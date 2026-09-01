"""Small, constrained blending helpers."""

from __future__ import annotations

import numpy as np


def blend_probability(reference: np.ndarray, branch: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return np.clip((1.0 - alpha) * reference + alpha * branch, 1e-6, 1 - 1e-6)


def disagreement_gate(
    direct: np.ndarray,
    pattern16: np.ndarray,
    auxiliary: np.ndarray,
    center: float = 0.02,
    sharpness: float = 40.0,
) -> np.ndarray:
    """Give more weight to an auxiliary branch when strong branches disagree."""
    disagreement = np.maximum(np.abs(pattern16 - direct), np.abs(auxiliary - pattern16))
    return 1.0 / (1.0 + np.exp(-sharpness * (disagreement - center)))
