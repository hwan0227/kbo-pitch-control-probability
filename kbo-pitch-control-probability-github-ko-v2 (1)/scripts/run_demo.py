"""Run a deterministic, data-free demonstration of the public pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitch_control.blending import blend_probability
from pitch_control.features import add_row_local_features
from pitch_control.pattern16 import (
    encode_pattern16,
    expected_control_probability,
    smoothed_success_rule,
)
from pitch_control.validation import summarize_folds


def synthetic_frame(rows: int = 2_000, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "balls_before": rng.integers(0, 4, rows),
            "strikes_before": rng.integers(0, 3, rows),
            "outs_before": rng.integers(0, 3, rows),
            "runner_on_1b": rng.integers(0, 2, rows),
            "runner_on_2b": rng.integers(0, 2, rows),
            "runner_on_3b": rng.integers(0, 2, rows),
            "score_diff_pitcher_team": rng.integers(-6, 7, rows),
            "pitcher_hand": rng.integers(0, 2, rows),
            "batter_hand": rng.integers(0, 2, rows),
            "asof_pitcher_n": rng.integers(0, 1_500, rows),
            "asof_pitcher_success_rate": rng.beta(40, 38, rows),
            "asof_pitcher_prev3_game_success_rate": rng.beta(15, 14, rows),
            "asof_batter_success_rate": rng.beta(40, 38, rows),
            "asof_pitcher_fastball_rate": rng.dirichlet([4, 3, 2], rows)[:, 0],
            "asof_pitcher_breaking_rate": rng.dirichlet([4, 3, 2], rows)[:, 1],
            "asof_pitcher_offspeed_rate": rng.dirichlet([4, 3, 2], rows)[:, 2],
        }
    )


def main() -> None:
    rng = np.random.default_rng(42)
    frame = add_row_local_features(synthetic_frame())

    event_probability = np.column_stack(
        [
            0.08 + 0.10 * frame["three_balls"],
            0.18 + 0.08 * frame["two_strikes"],
            0.24 + 0.09 * frame["three_balls"],
            0.32 + 0.10 * frame["two_strikes"],
        ]
    ).clip(0.01, 0.95)
    events = rng.binomial(1, event_probability)
    classes = encode_pattern16(events)

    latent = (
        -0.15
        + 0.8 * frame["asof_pitcher_success_rate"].to_numpy()
        - 0.15 * frame["three_balls"].to_numpy()
        + 0.10 * frame["two_strikes"].to_numpy()
        - 0.16 * events[:, 1]
        - 0.12 * events[:, 2]
    )
    true_probability = 1 / (1 + np.exp(-latent))
    y = rng.binomial(1, true_probability)

    rule = smoothed_success_rule(classes[:1_500], y[:1_500])
    pattern_prob = np.full((len(frame), 16), 0.15 / 15)
    pattern_prob[np.arange(len(frame)), classes] = 0.85
    p_pattern = expected_control_probability(pattern_prob, rule)

    p_reference = np.clip(
        0.35 + 0.30 * frame["asof_pitcher_success_rate"].to_numpy(), 0.02, 0.98
    )
    p_candidate = blend_probability(p_reference, p_pattern, alpha=0.10)
    folds = np.arange(len(frame)) % 5
    summary = summarize_folds(y, p_reference, p_candidate, folds)

    print("features:", frame.shape)
    print("pattern classes observed:", np.unique(classes).size)
    print("marginal fold summary:", summary.to_dict())


if __name__ == "__main__":
    main()
