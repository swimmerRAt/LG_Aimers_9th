"""Compact probability ensemble used by the offline submission artifact."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


CATEGORICAL_COLUMNS = ("top_bottom", "game_type", "base_state")
SMOOTHING_RATE_SPECS = (
    ("pitcher", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "asof_batter_n", "asof_batter_success_rate"),
)


class RateSmoothingFeatureBuilder:
    """Add empirical-Bayes rate features using a training-fold league prior."""

    def __init__(self, lambdas=(50.0, 200.0, 500.0, 1000.0)):
        self.lambdas = lambdas

    def fit(self, X, y):
        if not self.lambdas:
            raise ValueError("at least one smoothing lambda is required")
        values = np.asarray(self.lambdas, dtype=float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError("smoothing lambdas must be finite and positive")
        required = {
            column
            for _, count_column, rate_column in SMOOTHING_RATE_SPECS
            for column in (count_column, rate_column)
        }
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"smoothing input is missing columns: {missing}")
        target = np.asarray(y, dtype=float)
        self.prior_ = float(target.mean())
        self.raw_columns_ = list(X.columns)
        self.lambdas_ = tuple(float(value) for value in values)
        return self

    def transform(self, X):
        if not hasattr(self, "prior_"):
            raise ValueError("RateSmoothingFeatureBuilder must be fitted before transform")
        missing = [column for column in self.raw_columns_ if column not in X.columns]
        if missing:
            raise ValueError(f"smoothing input is missing columns: {missing}")
        result = X.loc[:, self.raw_columns_].copy()
        for prefix, count_column, rate_column in SMOOTHING_RATE_SPECS:
            count = pd.to_numeric(result[count_column], errors="coerce")
            rate = pd.to_numeric(result[rate_column], errors="coerce")
            valid = count.notna() & rate.notna() & (count >= 0.0)
            safe_count = count.where(valid, 0.0)
            safe_rate = rate.where(valid, self.prior_)
            for value in self.lambdas_:
                suffix = int(value) if float(value).is_integer() else str(value).replace(".", "p")
                result[f"{prefix}_success_rate_smooth_{suffix}"] = (
                    safe_count * safe_rate + value * self.prior_
                ) / (safe_count + value)
        return result


class OptimizedBaseballEnsemble(BaseEstimator, ClassifierMixin):
    """Blend a regularized HistGB model with a diverse ExtraTrees model."""

    def __init__(
        self,
        hist_weight: float = 0.45,
        n_estimators: int = 160,
        random_state: int = 42,
        calibration_slope: float = 1.0,
        calibration_intercept: float = 0.0,
        smoothing_lambdas=(),
    ):
        self.hist_weight = hist_weight
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.calibration_slope = calibration_slope
        self.calibration_intercept = calibration_intercept
        self.smoothing_lambdas = smoothing_lambdas

    def fit(self, X, y, sample_weight=None):
        raw_feature_columns = list(X.columns)
        if self.smoothing_lambdas:
            self.smoothing_builder_ = RateSmoothingFeatureBuilder(
                self.smoothing_lambdas
            ).fit(X, y)
            model_features = self.smoothing_builder_.transform(X)
        else:
            model_features = X
        feature_columns = list(model_features.columns)
        categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
        numeric = [column for column in feature_columns if column not in categorical]
        self.preprocessor_ = ColumnTransformer([
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "encode",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]),
                categorical,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ])
        transformed = self.preprocessor_.fit_transform(model_features, y)
        self.hist_model_ = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=self.random_state,
        ).fit(transformed, y, sample_weight=sample_weight)
        self.extra_model_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=16,
            min_samples_leaf=100,
            max_features=0.8,
            n_jobs=-1,
            random_state=self.random_state,
        ).fit(transformed, y, sample_weight=sample_weight)
        self.classes_ = np.asarray(self.hist_model_.classes_)
        self.feature_names_in_ = np.asarray(raw_feature_columns, dtype=object)
        return self

    @staticmethod
    def _positive_probability(model, transformed):
        classes = np.asarray(model.classes_)
        position = np.flatnonzero(classes == 1)
        if len(position) != 1:
            raise ValueError(f"class 1 not found exactly once in {classes.tolist()}")
        return model.predict_proba(transformed)[:, int(position[0])]

    def predict_proba(self, X):
        check_is_fitted(self, ["preprocessor_", "hist_model_", "extra_model_", "classes_"])
        model_features = self.smoothing_builder_.transform(X) if hasattr(
            self, "smoothing_builder_"
        ) else X
        transformed = self.preprocessor_.transform(model_features)
        hist = self._positive_probability(self.hist_model_, transformed)
        extra = self._positive_probability(self.extra_model_, transformed)
        positive = self.hist_weight * hist + (1.0 - self.hist_weight) * extra
        positive = np.clip(
            self.calibration_slope * positive + self.calibration_intercept,
            0.0,
            1.0,
        )
        return np.column_stack([1.0 - positive, positive])

    def feature_importance_frame(self):
        """Return fitted ExtraTrees impurity importance in descending order."""
        check_is_fitted(self, ["preprocessor_", "extra_model_"])
        names = self.preprocessor_.get_feature_names_out()
        importance = np.asarray(self.extra_model_.feature_importances_, dtype=float)
        if len(names) != len(importance):
            raise ValueError(
                f"feature name/importance mismatch: {len(names)} != {len(importance)}"
            )
        cleaned_names = np.asarray(
            [name.split("__", 1)[-1] for name in names],
            dtype=object,
        )
        order = np.argsort(-importance, kind="stable")
        return cleaned_names[order], importance[order]

    @property
    def feature_importance_source(self) -> str:
        return "ExtraTrees impurity importance (55% of probability ensemble)"
