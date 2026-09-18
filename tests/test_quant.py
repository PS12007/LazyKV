"""int8 packing for the warm/cold tier (rung 8): layout, round-trip error, and the axes of the scales."""

from __future__ import annotations

import pytest
import torch

from lazykv.quant import SCALE_DTYPE, pack, packed_pair_bytes, unpack

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BS, D = 64, 64


def _roundtrip(x: torch.Tensor) -> torch.Tensor:
    packed = pack(x)
    flat = packed.reshape(-1, packed.shape[-1])
    return unpack(flat, BS, D, x.dtype).reshape(x.shape)


def test_record_is_half_the_width_of_bf16_plus_the_scales() -> None:
    width = packed_pair_bytes(BS, D)
    payload = 2 * BS * D
    assert width == payload + 2 * SCALE_DTYPE.itemsize * (BS + D)
    # The capacity claim of rung 8: a little over half a bf16 pair, the overhead being the scales.
    assert 0.5 < width / (payload * 2) < 0.55


def test_pack_shapes_keep_the_leading_axes() -> None:
    x = torch.randn(5, 3, 2, BS, D, dtype=torch.bfloat16)
    packed = pack(x)
    assert packed.shape == (5, 3, packed_pair_bytes(BS, D))
    assert packed.dtype == torch.uint8


def test_roundtrip_error_is_within_one_level_of_the_group_range() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 2, BS, D, dtype=torch.bfloat16) * 3
    back = _roundtrip(x)
    assert back.dtype == x.dtype
    # Keys are quantized per channel (range over tokens), values per token (range over head_dim).
    k_range = (x[:, 0].amax(1) - x[:, 0].amin(1)).float().unsqueeze(1)
    v_range = (x[:, 1].amax(2) - x[:, 1].amin(2)).float().unsqueeze(2)
    level = torch.stack([k_range.expand_as(x[:, 0]), v_range.expand_as(x[:, 1])], dim=1) / 255
    # One level covers the rounding; the rest is the float16 scale and the bf16 step of the output.
    assert ((back.float() - x.float()).abs() <= level + back.float().abs() * 2**-7 + 1e-6).all()


def test_a_key_channel_outlier_does_not_degrade_its_neighbours() -> None:
    """Why keys are quantized per channel: one huge channel must not consume every channel's range."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, BS, D, dtype=torch.bfloat16)
    x[0, 0, :, 0] *= 500  # a single outlier channel, as KIVI describes for keys
    back = _roundtrip(x)
    quiet = (back[0, 0, :, 1:].float() - x[0, 0, :, 1:].float()).abs().max()
    assert quiet < 0.05  # a per-token scale would have to cover +-500 here and lose the quiet channels


def test_constant_group_is_exact() -> None:
    """A zero range must not divide by zero; any positive scale maps the group back through `zero`."""
    x = torch.full((2, 2, BS, D), 3.5, dtype=torch.bfloat16)
    assert torch.equal(_roundtrip(x), x)


def test_unpack_rejects_a_record_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="uint8 records"):
        unpack(torch.zeros(2, 7, dtype=torch.uint8), BS, D, torch.bfloat16)


def test_pack_rejects_a_tensor_that_is_not_a_kv_pair() -> None:
    with pytest.raises(ValueError, match="KV pair"):
        pack(torch.zeros(3, 3, BS, D, dtype=torch.bfloat16))


@cuda
def test_packing_on_the_gpu_agrees_with_the_host_to_within_a_level() -> None:
    """Rung 8 is not bit-exact, and this is where that starts.

    CPU and GPU float32 division disagree on the last bit, so a value sitting exactly on a rounding
    boundary lands one level either way and the packed bytes are not identical across devices. The
    dequantized values still agree to within that level. Rungs 6 and 7 could claim bit-identity
    with rung 5; rung 8 cannot, and the experiment must report agreement rather than assume it.
    """
    torch.manual_seed(0)
    x = torch.randn(3, 2, BS, D, dtype=torch.bfloat16)
    host = unpack(pack(x), BS, D, torch.float32)
    dev = unpack(pack(x.cuda()), BS, D, torch.float32).cpu()
    level = (x.float().amax(-1, keepdim=True) - x.float().amin(-1, keepdim=True)).abs().max() / 255
    assert (host - dev).abs().max() <= level
