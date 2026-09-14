"""KV geometry of candidate models, computed from their real config.json (brief section B3).

Downloads only config.json and repo file metadata (sizes), never weights.
Writes results/phase0/kv_memory_model/metrics.json plus a copy of each config for provenance.

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError  # noqa: E402

from harness.results import RESULTS_DIR, write_metrics  # noqa: E402

log = logging.getLogger("kv_model")

CONTEXTS = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072]
BUDGETS = [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625]


@dataclass(frozen=True)
class Candidate:
    key: str
    repo: str
    role: str
    # Ungated mirror with byte-identical config, used only if the official repo is gated
    # for this account. Which source was used is recorded next to every number.
    mirror: str | None = None


CANDIDATES = [
    Candidate("llama-3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", "primary (brief)", "unsloth/Llama-3.2-1B-Instruct"),
    Candidate("llama-3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "stress (brief)", "unsloth/Llama-3.2-3B-Instruct"),
    Candidate("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "candidate / substitute"),
    Candidate("phi-3-mini", "microsoft/Phi-3-mini-4k-instruct", "candidate (no GQA)"),
    Candidate("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "substitute (brief)"),
]


def fetch_config(c: Candidate, cfg_dir: Path) -> tuple[dict[str, Any], str, bool]:
    for repo, is_mirror in ((c.repo, False), (c.mirror, True)):
        if repo is None:
            continue
        try:
            path = hf_hub_download(repo, "config.json")
        except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as exc:
            log.warning("%s: config not accessible (%s)", repo, type(exc).__name__)
            continue
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        (cfg_dir / f"{c.key}.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        return cfg, repo, is_mirror
    raise SystemExit(f"could not fetch config for {c.key}")


def weight_bytes(repo: str) -> int | None:
    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError):
        return None
    sizes = [s.size for s in info.siblings or [] if s.rfilename.endswith(".safetensors") and s.size]
    return sum(sizes) if sizes else None


def geometry(cfg: dict[str, Any]) -> dict[str, Any]:
    # Phi-3 and Llama nest nothing; some newer configs put text params under text_config.
    cfg = cfg.get("text_config", cfg)
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
    n_layers = cfg["num_hidden_layers"]
    per_token_bf16 = 2 * n_layers * n_kv * head_dim * 2
    return {
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "head_dim_source": "config.head_dim" if cfg.get("head_dim") else "hidden_size / num_attention_heads",
        "gqa_group": n_heads // n_kv,
        "hidden_size": cfg["hidden_size"],
        "vocab_size": cfg["vocab_size"],
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "rope_scaling": cfg.get("rope_scaling"),
        "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "kv_bytes_per_token_bf16": per_token_bf16,
        "kv_bytes_per_token_int8": per_token_bf16 // 2,
        "kv_bytes_by_context_bf16": {str(n): per_token_bf16 * n for n in CONTEXTS},
        # Full-sequence logits at prefill: vocab x ctx x dtype. This is the chunked-prefill
        # trap from B3; 4 bytes covers the common float32 upcast of logits.
        "prefill_logits_bytes_64k": {
            "bf16": cfg["vocab_size"] * 65_536 * 2,
            "fp32": cfg["vocab_size"] * 65_536 * 4,
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_dir = RESULTS_DIR / "phase0" / "kv_memory_model"
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    vram_total = vram_free = None
    run1 = RESULTS_DIR / "phase0" / "feasibility" / "run_1" / "metrics.json"
    if run1.exists():
        sw = json.loads(run1.read_text(encoding="utf-8"))["system"]["software_gpu"]
        vram_total, vram_free = sw.get("cuda_mem_total_bytes"), sw.get("cuda_mem_free_bytes_at_start")

    models = []
    for c in CANDIDATES:
        cfg, source, is_mirror = fetch_config(c, cfg_dir)
        g = geometry(cfg)
        wb = weight_bytes(source)
        entry: dict[str, Any] = {**asdict(c), "config_source": source, "used_mirror": is_mirror, **g}
        entry["weights_bytes_safetensors"] = wb
        if wb is not None and vram_free is not None:
            entry["max_context_bf16_in_free_vram"] = max(0, vram_free - wb) // g["kv_bytes_per_token_bf16"]
        models.append(entry)
        log.info("%s: %d bytes/token (bf16) from %s", c.key, g["kv_bytes_per_token_bf16"], source)

    primary = next(m for m in models if m["key"] == "llama-3.2-1b")
    sweep_ctx = 32_768
    budget_sweep = [
        {"budget": b, "gpu_kv_bytes": int(b * primary["kv_bytes_per_token_bf16"] * sweep_ctx)} for b in BUDGETS
    ]
    write_metrics(
        out_dir,
        {
            "formula": "2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes",
            "contexts": CONTEXTS,
            "vram_total_bytes": vram_total,
            "vram_free_bytes_at_start": vram_free,
            "models": models,
            "primary_budget_sweep": {"model": primary["key"], "ctx_len": sweep_ctx, "rows": budget_sweep},
        },
    )


if __name__ == "__main__":
    main()
