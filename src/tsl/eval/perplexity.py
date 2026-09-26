from __future__ import annotations

import math


def loss_to_perplexity(mean_nll: float, *, max_nll: float = 20.0) -> float:
    # same cap as the training eval path; early losses blow up exp()
    return float(math.exp(min(float(mean_nll), max_nll)))
