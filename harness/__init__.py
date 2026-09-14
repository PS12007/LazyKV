"""Measurement harness shared by every LazyKV experiment.

Deliberately separate from the (future) ``lazykv`` runtime package: this code only
measures and records, it never manages KV state.
"""
