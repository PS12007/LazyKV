"""LazyKV: tiered KV-cache residency study on a single consumer GPU.

Rung 1 (full GPU residency) lives in lazykv/cache.py; the block pool and GPU-only
residency policies (rungs 2 and 3) in lazykv/blocks.py and lazykv/policies.py.
"""
