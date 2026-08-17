from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model.trackman_features import (
    PHYSICAL_COLUMNS,
    TrackmanMatchThresholds,
    build_pitcher_mapping,
    build_trackman_feature_lookup,
    high_confidence_mask,
    trackman_hand_code,
)


def main_pitcher_rows(pitcher_id: int, fastball: float, breaking: float) -> pd.DataFrame:
    count = 60
    return pd.DataFrame(
        {
            "season": 2019,
            "pitcher_id": pitcher_id,
            "pitcher_hand": 1,
            "asof_pitcher_pitchmix_n": np.arange(count),
            "asof_pitcher_fastball_rate": fastball,
            "asof_pitcher_breaking_rate": breaking,
            "asof_pitcher_offspeed_rate": 0.0,
        }
    )


def trackman_pitcher_rows(
    pitcher_id: int,
    fastball_count: int,
    breaking_count: int,
    speed: float,
    season: int = 2019,
) -> pd.DataFrame:
    groups = ["fastball"] * fastball_count + ["breaking"] * breaking_count
    frame = pd.DataFrame(
        {
            "season": season,
            "pitcher_trackman_id": pitcher_id,
            "pitcher_hand": "Right",
            "pitch_type_group": groups,
        }
    )
    for column in PHYSICAL_COLUMNS:
        frame[column] = speed
    return frame


class TrackmanFeatureTest(unittest.TestCase):
    def test_hand_codes_match_main_convention(self):
        values = trackman_hand_code(pd.Series(["Right", "Left", "Unknown"]))
        self.assertEqual(values.iloc[0], 1)
        self.assertEqual(values.iloc[1], 2)
        self.assertTrue(pd.isna(values.iloc[2]))

    def test_high_confidence_rule_requires_mutual_nearest(self):
        frame = pd.DataFrame(
            {
                "mutual_nearest": [True, False],
                "rate_cost": [0.01, 0.01],
                "count_cost": [0.1, 0.1],
                "season_cost": [0.1, 0.1],
                "margin": [0.02, 0.02],
                "main_history_n": [100, 100],
            }
        )
        self.assertEqual(high_confidence_mask(frame).tolist(), [True, False])

    def test_mapping_recovers_distinct_pitch_mix_profiles(self):
        main = pd.concat(
            [main_pitcher_rows(10, 0.7, 0.3), main_pitcher_rows(20, 0.3, 0.7)],
            ignore_index=True,
        )
        trackman = pd.concat(
            [
                trackman_pitcher_rows(100, 42, 18, 100.0),
                trackman_pitcher_rows(200, 18, 42, 110.0),
            ],
            ignore_index=True,
        )
        mapping = build_pitcher_mapping(main, trackman, cutoff_season=2020)
        accepted = mapping[mapping["high_confidence"]]
        self.assertEqual(dict(zip(accepted.pitcher_id, accepted.pitcher_trackman_id)), {10: 100, 20: 200})

    def test_physical_lookup_excludes_cutoff_and_future_seasons(self):
        prior = pd.concat(
            [
                trackman_pitcher_rows(100, 42, 18, 100.0),
                trackman_pitcher_rows(200, 18, 42, 110.0),
            ],
            ignore_index=True,
        )
        future = trackman_pitcher_rows(100, 50, 50, 500.0, season=2020)
        trackman = pd.concat([prior, future], ignore_index=True)
        mapping = pd.DataFrame(
            {
                "pitcher_id": [10],
                "pitcher_trackman_id": [100],
                "pitcher_hand": [1],
                "assignment_cost": [0.01],
                "margin": [0.10],
                "high_confidence": [True],
            }
        )
        lookup = build_trackman_feature_lookup(
            trackman, mapping, cutoff_season=2020, shrinkage=0.0
        )
        self.assertAlmostEqual(lookup.loc[0, "tm_log1p_history_n"], np.log1p(60))
        # The 2019 hand mean is 105, so pitcher 100's delta is -5. Future speed 500 is excluded.
        self.assertAlmostEqual(lookup.loc[0, "tm_rel_speed_delta"], -5.0)


if __name__ == "__main__":
    unittest.main()

