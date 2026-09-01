"""Row-local feature construction.

Every output row is a pure function of the same input row. Historical aggregates
must already have been computed using a strict cutoff before entering this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "score_diff_pitcher_team",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_batter_success_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
}


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def add_row_local_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with compact count, base, form, and pitch-mix features."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    out = frame.copy()
    balls = out["balls_before"].astype(int)
    strikes = out["strikes_before"].astype(int)

    out["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    out["count_diff"] = strikes - balls
    out["two_strikes"] = (strikes == 2).astype("int8")
    out["three_balls"] = (balls == 3).astype("int8")
    out["full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
    out["first_pitch"] = ((balls == 0) & (strikes == 0)).astype("int8")
    out["ahead_in_count"] = (strikes > balls).astype("int8")

    out["base_state"] = (
        out["runner_on_1b"].astype(int)
        + 2 * out["runner_on_2b"].astype(int)
        + 4 * out["runner_on_3b"].astype(int)
    )
    out["base_out_state"] = out["base_state"] * 3 + out["outs_before"].astype(int)
    out["scoring_position"] = (
        (out["runner_on_2b"] > 0) | (out["runner_on_3b"] > 0)
    ).astype("int8")
    out["abs_score_diff"] = out["score_diff_pitcher_team"].abs()
    out["same_hand"] = (out["pitcher_hand"] == out["batter_hand"]).astype("int8")

    out["asof_pitcher_n_log"] = np.log1p(out["asof_pitcher_n"].clip(lower=0))
    out["form_gap_3"] = (
        out["asof_pitcher_prev3_game_success_rate"]
        - out["asof_pitcher_success_rate"]
    )
    out["matchup_gap"] = (
        out["asof_pitcher_success_rate"] - out["asof_batter_success_rate"]
    )

    mix = out[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].to_numpy(float)
    mix = np.clip(mix, 0.0, None)
    row_sum = mix.sum(axis=1, keepdims=True)
    mix = np.divide(mix, row_sum, out=np.full_like(mix, 1 / 3), where=row_sum > 0)
    out["pitchmix_entropy"] = _entropy(mix)
    out["pitchmix_max"] = mix.max(axis=1)
    return out
