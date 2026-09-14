"""LazyKV: tiered KV-cache residency study on a single consumer GPU.

Phase 1 contains only the full-residency reference (policy ladder rung 1) and the
measurement plumbing every later policy reuses.
"""
