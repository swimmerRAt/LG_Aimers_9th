"""Leakage-safe probabilistic pitcher mapping and Trackman feature aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PITCH_GROUPS = ("fastball", "breaking", "offspeed")
PHYSICAL_COLUMNS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
MAIN_RATE_COLUMNS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


@dataclass(frozen=True)
class TrackmanMatchThresholds:
    """Conservative thresholds selected from pre-2024 mapping stability."""

    max_rate_cost: float = 0.025
    max_count_cost: float = 0.5
    max_season_cost: float = 0.4
    min_margin: float = 0.01
    min_history: float = 50.0


def trackman_hand_code(values) -> pd.Series:
    """Convert Trackman hand labels to the main-table numeric convention."""
    series = pd.Series(values, copy=False)
    return series.map({"Right": 1, "Left": 2}).astype("Int64")


def high_confidence_mask(
    mapping: pd.DataFrame,
    thresholds: TrackmanMatchThresholds = TrackmanMatchThresholds(),
) -> pd.Series:
    """Return the fixed, target-free adoption mask for entity matches."""
    return (
        mapping["mutual_nearest"].astype(bool)
        & (mapping["rate_cost"] <= thresholds.max_rate_cost)
        & (mapping["count_cost"] <= thresholds.max_count_cost)
        & (mapping["season_cost"] <= thresholds.max_season_cost)
        & (mapping["margin"] >= thresholds.min_margin)
        & (mapping["main_history_n"] >= thresholds.min_history)
    )


def _mode(series: pd.Series):
    modes = series.dropna().mode()
    return modes.iloc[0] if len(modes) else np.nan


def build_pitcher_mapping(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    cutoff_season: int,
    thresholds: TrackmanMatchThresholds = TrackmanMatchThresholds(),
) -> pd.DataFrame:
    """Match pitchers using only rows strictly before ``cutoff_season``.

    The cost combines handedness-constrained pitch-mix rates, total history size,
    and per-season activity. Hungarian assignment enforces a one-to-one mapping;
    the returned high-confidence flag additionally requires mutual nearest
    neighbors and a sufficient first-versus-second candidate margin.
    """
    main_history = main.loc[main["season"] < cutoff_season].copy()
    track_history = trackman.loc[trackman["season"] < cutoff_season].copy()
    output_columns = [
        "cutoff_season",
        "pitcher_id",
        "pitcher_trackman_id",
        "pitcher_hand",
        "rate_cost",
        "count_cost",
        "season_cost",
        "assignment_cost",
        "margin",
        "mutual_nearest",
        "main_history_n",
        "trackman_history_n",
        "high_confidence",
    ]
    if main_history.empty or track_history.empty:
        return pd.DataFrame(columns=output_columns)

    main_history = main_history.reset_index(drop=False).rename(columns={"index": "_row_order"})
    track_history = track_history.copy()
    track_history["_hand_code"] = trackman_hand_code(track_history["pitcher_hand"])

    history_count = pd.to_numeric(
        main_history["asof_pitcher_pitchmix_n"], errors="coerce"
    ).fillna(-1.0)
    main_history["_history_count"] = history_count
    last = (
        main_history.sort_values(
            ["pitcher_id", "_history_count", "_row_order"], kind="stable"
        )
        .groupby("pitcher_id", sort=False)
        .tail(1)
        .set_index("pitcher_id")
    )
    seasons = list(range(int(main_history["season"].min()), cutoff_season))
    main_season = (
        main_history.groupby(["pitcher_id", "season"]).size().unstack(fill_value=0)
        .reindex(index=last.index, columns=seasons, fill_value=0)
    )

    track_counts = (
        track_history.groupby(["pitcher_trackman_id", "pitch_type_group"])
        .size()
        .unstack(fill_value=0)
    )
    for group in PITCH_GROUPS:
        if group not in track_counts:
            track_counts[group] = 0
    track_counts = track_counts.loc[:, list(PITCH_GROUPS)]
    track_denominator = track_counts.sum(axis=1)
    track_rates = track_counts.div(
        track_denominator.replace(0, np.nan), axis=0
    ).fillna(0.0)
    track_season = (
        track_history.groupby(["pitcher_trackman_id", "season"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=track_counts.index, columns=seasons, fill_value=0)
    )
    track_hands = (
        track_history.groupby("pitcher_trackman_id")["_hand_code"]
        .agg(_mode)
        .reindex(track_counts.index)
    )

    rows: list[dict] = []
    for hand in (1, 2):
        main_hand = last[pd.to_numeric(last["pitcher_hand"], errors="coerce") == hand]
        track_ids = track_hands.index[track_hands == hand]
        if main_hand.empty or len(track_ids) == 0:
            continue

        main_rates = (
            main_hand.loc[:, list(MAIN_RATE_COLUMNS)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
        candidate_rates = track_rates.loc[track_ids].to_numpy(float)
        main_n = (
            pd.to_numeric(main_hand["asof_pitcher_pitchmix_n"], errors="coerce")
            .fillna(-1.0)
            .to_numpy(float)
            + 1.0
        )
        track_n = track_counts.loc[track_ids].sum(axis=1).to_numpy(float)
        main_activity = main_season.loc[main_hand.index].to_numpy(float)
        track_activity = track_season.loc[track_ids].to_numpy(float)

        rate_cost = (
            np.abs(main_rates[:, None, :] - candidate_rates[None, :, :]).sum(axis=2)
            / 2.0
        )
        count_cost = np.abs(main_n[:, None] - track_n[None, :]) / np.maximum(
            50.0, main_n[:, None]
        )
        season_cost = np.mean(
            np.abs(main_activity[:, None, :] - track_activity[None, :, :])
            / np.maximum(25.0, main_activity[:, None, :]),
            axis=2,
        )
        assignment_cost = (
            rate_cost
            + 0.15 * np.minimum(count_cost, 2.0)
            + 0.35 * np.minimum(season_cost, 2.0)
        )
        assigned_main, assigned_track = linear_sum_assignment(assignment_cost)
        nearest_by_main = np.argmin(assignment_cost, axis=1)
        nearest_by_track = np.argmin(assignment_cost, axis=0)
        sorted_cost = np.sort(assignment_cost, axis=1)
        margins = (
            sorted_cost[:, 1] - sorted_cost[:, 0]
            if assignment_cost.shape[1] > 1
            else np.full(assignment_cost.shape[0], np.inf)
        )

        for main_position, track_position in zip(assigned_main, assigned_track):
            rows.append(
                {
                    "cutoff_season": int(cutoff_season),
                    "pitcher_id": main_hand.index[main_position],
                    "pitcher_trackman_id": track_ids[track_position],
                    "pitcher_hand": hand,
                    "rate_cost": float(rate_cost[main_position, track_position]),
                    "count_cost": float(count_cost[main_position, track_position]),
                    "season_cost": float(season_cost[main_position, track_position]),
                    "assignment_cost": float(
                        assignment_cost[main_position, track_position]
                    ),
                    "margin": float(margins[main_position]),
                    "mutual_nearest": bool(
                        nearest_by_main[main_position] == track_position
                        and nearest_by_track[track_position] == main_position
                    ),
                    "main_history_n": float(main_n[main_position]),
                    "trackman_history_n": float(track_n[track_position]),
                }
            )

    mapping = pd.DataFrame(rows)
    if mapping.empty:
        return pd.DataFrame(columns=output_columns)
    mapping["high_confidence"] = high_confidence_mask(mapping, thresholds)
    return mapping.loc[:, output_columns].sort_values(
        ["high_confidence", "assignment_cost"], ascending=[False, True]
    ).reset_index(drop=True)


def trackman_feature_columns() -> list[str]:
    columns = [
        "tm_is_mapped",
        "tm_match_quality",
        "tm_log1p_history_n",
    ]
    columns.extend(f"tm_{column}_delta" for column in PHYSICAL_COLUMNS)
    columns.extend(f"tm_{column}_std_delta" for column in PHYSICAL_COLUMNS)
    columns.extend(f"tm_{group}_rate_delta" for group in PITCH_GROUPS)
    return columns


def build_trackman_feature_lookup(
    trackman: pd.DataFrame,
    mapping: pd.DataFrame,
    cutoff_season: int,
    shrinkage: float = 200.0,
) -> pd.DataFrame:
    """Aggregate prior Trackman rows for accepted mappings with hand shrinkage."""
    feature_columns = trackman_feature_columns()
    accepted = mapping.loc[mapping["high_confidence"].astype(bool)].copy()
    history = trackman.loc[trackman["season"] < cutoff_season].copy()
    if accepted.empty or history.empty:
        return pd.DataFrame(columns=["pitcher_id", *feature_columns])
    history["_hand_code"] = trackman_hand_code(history["pitcher_hand"])

    pitcher_group = history.groupby("pitcher_trackman_id", sort=False)
    hand_group = history.groupby("_hand_code", sort=False)
    pitcher_mean = pitcher_group[list(PHYSICAL_COLUMNS)].mean()
    pitcher_std = pitcher_group[list(PHYSICAL_COLUMNS)].std(ddof=0).fillna(0.0)
    hand_mean = hand_group[list(PHYSICAL_COLUMNS)].mean()
    hand_std = hand_group[list(PHYSICAL_COLUMNS)].std(ddof=0).fillna(0.0)
    pitcher_n = pitcher_group.size().astype(float)

    pitcher_pitch = (
        history.groupby(["pitcher_trackman_id", "pitch_type_group"])
        .size()
        .unstack(fill_value=0)
    )
    hand_pitch = (
        history.groupby(["_hand_code", "pitch_type_group"])
        .size()
        .unstack(fill_value=0)
    )
    for group in PITCH_GROUPS:
        if group not in pitcher_pitch:
            pitcher_pitch[group] = 0
        if group not in hand_pitch:
            hand_pitch[group] = 0
    pitcher_pitch = pitcher_pitch.loc[:, list(PITCH_GROUPS)]
    hand_pitch = hand_pitch.loc[:, list(PITCH_GROUPS)]
    pitcher_rates = pitcher_pitch.div(
        pitcher_pitch.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0.0)
    hand_rates = hand_pitch.div(
        hand_pitch.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0.0)

    rows = []
    for match in accepted.itertuples(index=False):
        track_id = match.pitcher_trackman_id
        hand = int(match.pitcher_hand)
        if track_id not in pitcher_mean.index or hand not in hand_mean.index:
            continue
        n = float(pitcher_n.loc[track_id])
        reliability = n / (n + float(shrinkage))
        quality = float(match.margin) / (
            float(match.margin) + float(match.assignment_cost) + 1e-12
        )
        row = {
            "pitcher_id": match.pitcher_id,
            "tm_is_mapped": 1.0,
            "tm_match_quality": quality,
            "tm_log1p_history_n": float(np.log1p(n)),
        }
        for column in PHYSICAL_COLUMNS:
            row[f"tm_{column}_delta"] = reliability * float(
                pitcher_mean.loc[track_id, column] - hand_mean.loc[hand, column]
            )
            row[f"tm_{column}_std_delta"] = reliability * float(
                pitcher_std.loc[track_id, column] - hand_std.loc[hand, column]
            )
        for group in PITCH_GROUPS:
            row[f"tm_{group}_rate_delta"] = reliability * float(
                pitcher_rates.loc[track_id, group] - hand_rates.loc[hand, group]
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=["pitcher_id", *feature_columns])


def add_temporal_trackman_features(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    thresholds: TrackmanMatchThresholds = TrackmanMatchThresholds(),
    shrinkage: float = 200.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add fixed prior-season features to every main-table row."""
    result = main.copy()
    feature_columns = trackman_feature_columns()
    for column in feature_columns:
        result[column] = 0.0

    mappings = []
    for season in sorted(int(value) for value in result["season"].unique()):
        mapping = build_pitcher_mapping(result, trackman, season, thresholds)
        mappings.append(mapping)
        lookup = build_trackman_feature_lookup(
            trackman, mapping, season, shrinkage=shrinkage
        )
        if lookup.empty:
            continue
        mask = result["season"] == season
        joined = result.loc[mask, ["pitcher_id"]].merge(
            lookup, on="pitcher_id", how="left", validate="many_to_one"
        )
        result.loc[mask, feature_columns] = (
            joined[feature_columns].fillna(0.0).to_numpy(float)
        )
    diagnostics = (
        pd.concat(mappings, ignore_index=True)
        if mappings
        else pd.DataFrame()
    )
    return result, diagnostics
