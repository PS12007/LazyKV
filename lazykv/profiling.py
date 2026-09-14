"""Per-layer decode time split via CUDA events on forward hooks.

Phase 0 left one number open: the real compute window a layer-ahead prefetch can overlap.
It used GPU attention alone as a lower bound. This module measures the whole layer, split
into attention kernel / rest of attention (projections, RoPE, cache write) / MLP / other
(norms, residuals).

Hooks add host overhead, so profiled steps are reported separately from timed steps and
the overhead itself is recorded rather than hidden.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel

from lazykv.attention import kernel_timer


@dataclass
class _Span:
    start: torch.cuda.Event
    end: torch.cuda.Event


@dataclass
class LayerProfiler:
    model: PreTrainedModel
    _spans: dict[str, list[_Span]] = field(default_factory=lambda: defaultdict(list))
    _open: dict[str, torch.cuda.Event] = field(default_factory=dict)
    _handles: list[RemovableHandle] = field(default_factory=list)

    def _pre(self, key: str):  # noqa: ANN202
        def hook(module, args, kwargs=None):  # noqa: ANN001, ANN202
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._open[key] = ev

        return hook

    def _post(self, key: str):  # noqa: ANN202
        def hook(module, args, output, kwargs=None):  # noqa: ANN001, ANN202
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._spans[key].append(_Span(self._open.pop(key), ev))

        return hook

    def __enter__(self) -> LayerProfiler:
        for i, layer in enumerate(self.model.model.layers):
            for name, mod in (("layer", layer), ("self_attn", layer.self_attn), ("mlp", layer.mlp)):
                key = f"{i}.{name}"
                self._handles.append(mod.register_forward_pre_hook(self._pre(key), with_kwargs=True))
                self._handles.append(mod.register_forward_hook(self._post(key), with_kwargs=True))
        kernel_timer().reset()
        kernel_timer().enabled = True
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        kernel_timer().enabled = False

    def reset(self) -> None:
        self._spans.clear()
        self._open.clear()
        kernel_timer().reset()

    def collect(self) -> dict[int, dict[str, float]]:
        """Synchronize, then return mean seconds per call for each layer component."""
        torch.cuda.synchronize()
        acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for key, spans in self._spans.items():
            idx, name = key.split(".")
            acc[int(idx)][name] += [s.start.elapsed_time(s.end) / 1e3 for s in spans]
        for layer_idx, start, end in kernel_timer().events:
            acc[layer_idx]["kernel"].append(start.elapsed_time(end) / 1e3)
        out: dict[int, dict[str, float]] = {}
        for idx, parts in acc.items():
            mean = {k: sum(v) / len(v) for k, v in parts.items() if v}
            layer = mean.get("layer", 0.0)
            attn = mean.get("self_attn", 0.0)
            mlp = mean.get("mlp", 0.0)
            kernel = mean.get("kernel", 0.0)
            out[idx] = {
                "layer_s": layer,
                "attention_kernel_s": kernel,
                "attention_other_s": attn - kernel,
                "mlp_s": mlp,
                "other_s": layer - attn - mlp,
            }
        return out
