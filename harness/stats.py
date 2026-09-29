"""Re-export of lazykv.stats, so the benchmark scripts keep one import path for their statistics.

The module lives in the installable package because lazykv.quality needs its bootstrap CI, and
an installed lazykv cannot import this repo-only harness.
"""

from lazykv.stats import (  # noqa: F401
    OverheadFit,
    Summary,
    bootstrap_mean_ci,
    fit_overhead,
    knee_size,
    overlap_fraction,
    paired_ratio_ci,
    prompts_to_resolve,
    quantile,
    sign_test_p,
    summarize,
)
