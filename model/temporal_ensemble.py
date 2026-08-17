"""Temporal-window ensemble built from the current HistGB + ExtraTrees model."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from model.ensemble import OptimizedBaseballEnsemble


COMPONENT_NAMES = ("full", "recent_3", "recent_2", "time_weighted")


class TemporalWindowEnsemble(BaseEstimator, ClassifierMixin):
    """Average identical base models trained with different temporal views."""

    def __init__(
        self,
        component_weights=(0.25, 0.25, 0.25, 0.25),
        time_decay: float = 0.8,
        hist_weight: float = 0.45,
        n_estimators: int = 160,
        random_state: int = 42,
        smoothing_lambdas=(),
    ):
        self.component_weights = component_weights
        self.time_decay = time_decay
        self.hist_weight = hist_weight
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.smoothing_lambdas = smoothing_lambdas

    @staticmethod
    def _validated_component_weights(weights) -> np.ndarray:
        values = np.asarray(weights, dtype=float)
        if values.shape != (len(COMPONENT_NAMES),):
            raise ValueError(f"component_weights must have {len(COMPONENT_NAMES)} values")
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("component_weights must be finite and non-negative")
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("component_weights must have a positive sum")
        return values / total

    @staticmethod
    def component_masks(seasons, latest_season: int) -> dict[str, np.ndarray]:
        season_values = np.asarray(seasons)
        return {
            "full": np.ones(len(season_values), dtype=bool),
            "recent_3": season_values >= latest_season - 2,
            "recent_2": season_values >= latest_season - 1,
            "time_weighted": np.ones(len(season_values), dtype=bool),
        }

    @staticmethod
    def temporal_sample_weight(seasons, latest_season: int, decay: float) -> np.ndarray:
        if not 0.0 < float(decay) <= 1.0:
            raise ValueError("time_decay must be inside (0, 1]")
        age = latest_season - np.asarray(seasons, dtype=float)
        if (age < 0).any():
            raise ValueError("training seasons cannot be later than latest_season")
        return np.power(float(decay), age)

    def _make_base_model(self, component_index: int) -> OptimizedBaseballEnsemble:
        return OptimizedBaseballEnsemble(
            hist_weight=self.hist_weight,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            smoothing_lambdas=self.smoothing_lambdas,
        )

    def fit(self, X, y):
        if "season" not in X.columns:
            raise ValueError("TemporalWindowEnsemble requires the season feature")
        seasons = pd.to_numeric(X["season"], errors="coerce").to_numpy()
        if not np.isfinite(seasons).all():
            raise ValueError("season must contain only finite numeric values")
        latest_season = int(np.max(seasons))
        weights = self._validated_component_weights(self.component_weights)
        masks = self.component_masks(seasons, latest_season)
        target = np.asarray(y)

        self.models_ = {}
        for index, name in enumerate(COMPONENT_NAMES):
            if weights[index] == 0.0:
                continue
            mask = masks[name]
            if not mask.any():
                raise ValueError(f"temporal component {name} has no training rows")
            sample_weight = None
            if name == "time_weighted":
                sample_weight = self.temporal_sample_weight(
                    seasons[mask], latest_season, self.time_decay
                )
            model = self._make_base_model(index)
            model.fit(X.loc[mask], target[mask], sample_weight=sample_weight)
            self.models_[name] = model

        self.component_weights_ = weights
        self.latest_season_ = latest_season
        self.classes_ = np.asarray([0, 1])
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def predict_component_probabilities(self, X) -> dict[str, np.ndarray]:
        check_is_fitted(self, ["models_", "component_weights_", "classes_"])
        return {
            name: np.asarray(self.models_[name].predict_proba(X)[:, 1], dtype=float)
            for name in self.models_
        }

    def predict_proba(self, X):
        component = self.predict_component_probabilities(X)
        positive = np.zeros(len(X), dtype=float)
        for weight, name in zip(self.component_weights_, COMPONENT_NAMES):
            if weight > 0.0:
                positive += float(weight) * component[name]
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def feature_importance_frame(self):
        check_is_fitted(self, ["models_", "component_weights_"])
        combined = defaultdict(float)
        for weight, name in zip(self.component_weights_, COMPONENT_NAMES):
            if weight == 0.0:
                continue
            feature_names, importance = self.models_[name].feature_importance_frame()
            for feature, value in zip(feature_names, importance):
                combined[str(feature)] += float(weight) * float(value)
        names = np.asarray(list(combined), dtype=object)
        values = np.asarray([combined[name] for name in names], dtype=float)
        total = float(values.sum())
        if total > 0.0:
            values /= total
        order = np.argsort(-values, kind="stable")
        return names[order], values[order]

    @property
    def feature_importance_source(self) -> str:
        return "Temporal ExtraTrees weighted impurity importance"
