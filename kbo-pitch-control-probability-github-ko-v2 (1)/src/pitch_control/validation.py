"""Brier scoring and stability gates used in the experiment loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def brier_skill_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Return the competition-style Brier skill score on a 100,000 scale."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and probability must have equal shapes")
    base_rate = float(y.mean())
    denominator = base_rate * (1.0 - base_rate)
    if denominator <= 0:
        raise ValueError("both target classes are required")
    brier = float(np.mean((y - p) ** 2))
    return 100_000.0 * (1.0 - brier / denominator)


def marginal_gain(
    y_true: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    return brier_skill_score(y_true, candidate) - brier_skill_score(y_true, reference)


@dataclass(frozen=True)
class FoldSummary:
    gain: float
    positive_folds: int
    worst_fold: float
    fold_sd: float
    fold_gains: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_folds(
    y_true: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    fold: np.ndarray,
) -> FoldSummary:
    fold = np.asarray(fold)
    gains = []
    for value in np.unique(fold):
        mask = fold == value
        gains.append(marginal_gain(y_true[mask], reference[mask], candidate[mask]))
    gains_array = np.asarray(gains, dtype=float)
    return FoldSummary(
        gain=marginal_gain(y_true, reference, candidate),
        positive_folds=int((gains_array > 0).sum()),
        worst_fold=float(gains_array.min()),
        fold_sd=float(gains_array.std()),
        fold_gains=tuple(float(x) for x in gains_array),
    )


def passes_stability_gate(
    summary: FoldSummary,
    minimum_gain: float = 4.0,
    minimum_positive_folds: int = 4,
    minimum_worst_fold: float = -0.5,
) -> bool:
    return (
        summary.gain >= minimum_gain
        and summary.positive_folds >= minimum_positive_folds
        and summary.worst_fold > minimum_worst_fold
    )
