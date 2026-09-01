"""Four-event joint target used by the Pattern16 branch."""

from __future__ import annotations

import numpy as np


EVENT_ORDER = ("reverse", "middle", "ball", "strike")
BIT_WEIGHTS = np.array([1, 2, 4, 8], dtype=np.int16)


def encode_pattern16(events: np.ndarray) -> np.ndarray:
    """Encode an ``(n, 4)`` Boolean event matrix into classes 0..15."""
    values = np.asarray(events)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("events must have shape (n_rows, 4)")
    if not np.isin(values, [0, 1, False, True]).all():
        raise ValueError("events must be binary")
    return (values.astype(np.int16) * BIT_WEIGHTS).sum(axis=1)


def decode_pattern16(classes: np.ndarray) -> np.ndarray:
    """Decode classes 0..15 into ``reverse, middle, ball, strike`` bits."""
    classes = np.asarray(classes, dtype=np.int16)
    if ((classes < 0) | (classes > 15)).any():
        raise ValueError("Pattern16 class must be between 0 and 15")
    return ((classes[:, None] & BIT_WEIGHTS[None, :]) > 0).astype(np.int8)


def smoothed_success_rule(
    classes: np.ndarray,
    success: np.ndarray,
    smoothing: float = 100.0,
) -> np.ndarray:
    """Estimate ``P(success | pattern)`` with global-rate shrinkage."""
    classes = np.asarray(classes, dtype=np.int16)
    success = np.asarray(success, dtype=float)
    if classes.shape[0] != success.shape[0]:
        raise ValueError("classes and success must have equal length")
    global_rate = float(success.mean())
    count = np.bincount(classes, minlength=16).astype(float)
    total = np.bincount(classes, weights=success, minlength=16).astype(float)
    return (total + smoothing * global_rate) / (count + smoothing)


def expected_control_probability(
    pattern_probabilities: np.ndarray,
    success_rule: np.ndarray,
) -> np.ndarray:
    """Marginalize the 16-class distribution into a control probability."""
    probabilities = np.asarray(pattern_probabilities, dtype=float)
    rule = np.asarray(success_rule, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 16:
        raise ValueError("pattern_probabilities must have shape (n_rows, 16)")
    if rule.shape != (16,):
        raise ValueError("success_rule must have shape (16,)")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability row must sum to one")
    return probabilities @ rule
