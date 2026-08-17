"""Leakage-safe monthly rate forecasting for event-level probability models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _logit(probability) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


class MonthlyRateTimeSeries:
    """Ridge dynamic regression over monthly aggregate target logits.

    The model uses a linear time trend plus optional annual Fourier harmonics.
    It is fitted only on observed training months and can forecast a future
    ``season``/``game_month`` pair without reading any other test row.
    """

    def __init__(
        self,
        harmonic_order: int = 1,
        ridge: float = 1.0,
        recency_decay_per_year: float = 1.0,
        prior_strength: float = 1000.0,
    ):
        self.harmonic_order = harmonic_order
        self.ridge = ridge
        self.recency_decay_per_year = recency_decay_per_year
        self.prior_strength = prior_strength

    @staticmethod
    def _validated_time(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = sorted({"season", "game_month"} - set(frame.columns))
        if missing:
            raise ValueError(f"monthly time-series input is missing columns: {missing}")
        season = pd.to_numeric(frame["season"], errors="coerce").to_numpy(float)
        month = pd.to_numeric(frame["game_month"], errors="coerce").to_numpy(float)
        valid = (
            np.isfinite(season)
            & np.isfinite(month)
            & np.isclose(season, np.round(season))
            & np.isclose(month, np.round(month))
            & (month >= 1)
            & (month <= 12)
        )
        if not valid.all():
            raise ValueError("season and game_month must be finite integers; month must be 1..12")
        return season.astype(int), month.astype(int)

    def _design(self, season: np.ndarray, month: np.ndarray) -> np.ndarray:
        month_index = season * 12 + month
        trend_years = (month_index - self.center_month_index_) / 12.0
        columns = [np.ones(len(month_index)), trend_years]
        angle = 2.0 * np.pi * month / 12.0
        for order in range(1, self.harmonic_order_ + 1):
            columns.extend([np.sin(order * angle), np.cos(order * angle)])
        return np.column_stack(columns)

    def fit(self, frame: pd.DataFrame, target) -> "MonthlyRateTimeSeries":
        harmonic_order = int(self.harmonic_order)
        if harmonic_order < 0 or harmonic_order > 3:
            raise ValueError("harmonic_order must be between 0 and 3")
        ridge = float(self.ridge)
        decay = float(self.recency_decay_per_year)
        prior_strength = float(self.prior_strength)
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        if not np.isfinite(decay) or not 0.0 < decay <= 1.0:
            raise ValueError("recency_decay_per_year must be inside (0, 1]")
        if not np.isfinite(prior_strength) or prior_strength < 0.0:
            raise ValueError("prior_strength must be finite and non-negative")

        season, month = self._validated_time(frame)
        truth = np.asarray(target, dtype=float)
        if truth.shape != (len(frame),) or not np.isin(truth, [0.0, 1.0]).all():
            raise ValueError("target must be a one-dimensional binary array")
        aggregate = pd.DataFrame(
            {"season": season, "game_month": month, "target": truth}
        ).groupby(["season", "game_month"], sort=True)["target"].agg(
            successes="sum", rows="size"
        ).reset_index()
        if len(aggregate) < 3:
            raise ValueError("at least three observed months are required")

        self.harmonic_order_ = harmonic_order
        self.center_month_index_ = int(
            np.max(aggregate["season"].to_numpy() * 12 + aggregate["game_month"].to_numpy())
        )
        self.global_rate_ = float(truth.mean())
        self.global_logit_ = float(_logit([self.global_rate_])[0])
        smoothed_rate = (
            aggregate["successes"].to_numpy(float) + prior_strength * self.global_rate_
        ) / (aggregate["rows"].to_numpy(float) + prior_strength)
        response = _logit(smoothed_rate)
        design = self._design(
            aggregate["season"].to_numpy(int),
            aggregate["game_month"].to_numpy(int),
        )
        age_years = (
            self.center_month_index_
            - (aggregate["season"].to_numpy() * 12 + aggregate["game_month"].to_numpy())
        ) / 12.0
        weights = aggregate["rows"].to_numpy(float) * np.power(decay, age_years)
        weights /= weights.mean()
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(
            design.T @ (weights[:, None] * design) + penalty,
            design.T @ (weights * response),
        )
        self.monthly_training_frame_ = aggregate
        return self

    def predict_logit(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise ValueError("MonthlyRateTimeSeries must be fitted before prediction")
        season, month = self._validated_time(frame)
        return self._design(season, month) @ self.coef_

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        logits = self.predict_logit(frame)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))

    def predict_logit_offset(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the forecast deviation from the training-period global prior."""
        return self.predict_logit(frame) - self.global_logit_
