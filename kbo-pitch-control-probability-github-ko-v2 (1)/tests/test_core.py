import unittest

import numpy as np
import pandas as pd

from pitch_control.features import add_row_local_features
from pitch_control.pattern16 import decode_pattern16, encode_pattern16, expected_control_probability
from pitch_control.validation import brier_skill_score


class Pattern16Test(unittest.TestCase):
    def test_roundtrip(self):
        events = np.array([[0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
        classes = encode_pattern16(events)
        np.testing.assert_array_equal(classes, [0, 5, 15])
        np.testing.assert_array_equal(decode_pattern16(classes), events)

    def test_expectation(self):
        probability = np.zeros((2, 16))
        probability[0, 0] = 1
        probability[1, 15] = 1
        rule = np.linspace(0.1, 0.9, 16)
        np.testing.assert_allclose(expected_control_probability(probability, rule), [0.1, 0.9])


class ValidationTest(unittest.TestCase):
    def test_perfect_brier_skill(self):
        y = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(brier_skill_score(y, y), 100_000.0)


class RowLocalTest(unittest.TestCase):
    def test_row_order_does_not_change_features(self):
        frame = pd.DataFrame(
            {
                "balls_before": [0, 3], "strikes_before": [0, 2], "outs_before": [0, 2],
                "runner_on_1b": [0, 1], "runner_on_2b": [0, 0], "runner_on_3b": [0, 1],
                "score_diff_pitcher_team": [0, -2], "pitcher_hand": [0, 1], "batter_hand": [1, 1],
                "asof_pitcher_n": [10, 50], "asof_pitcher_success_rate": [0.5, 0.6],
                "asof_pitcher_prev3_game_success_rate": [0.55, 0.58],
                "asof_batter_success_rate": [0.48, 0.62],
                "asof_pitcher_fastball_rate": [0.5, 0.4],
                "asof_pitcher_breaking_rate": [0.3, 0.4],
                "asof_pitcher_offspeed_rate": [0.2, 0.2],
            }, index=[10, 20]
        )
        expected = add_row_local_features(frame).sort_index()
        actual = add_row_local_features(frame.iloc[::-1]).sort_index()
        pd.testing.assert_frame_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
