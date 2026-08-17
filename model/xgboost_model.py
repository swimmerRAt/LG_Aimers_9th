"""XGBoost probability candidate for local baseball-model comparison."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier


CATEGORICAL_COLUMNS = ("top_bottom", "game_type", "base_state")


class XGBoostProbabilityModel(BaseEstimator, ClassifierMixin):
    """One-hot encode low-cardinality categories and fit binary XGBoost.

    The validation metric is RMSE. Because the target is binary, RMSE squared
    is exactly the Brier score, so both metrics select the same iteration.
    """

    def __init__(
        self,
        n_estimators: int = 2000,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        min_child_weight: float = 100.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 15.0,
        max_bin: int = 256,
        early_stopping_rounds: int | None = 100,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.max_bin = max_bin
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_preprocessor(self, columns) -> ColumnTransformer:
        columns = list(columns)
        categorical = [column for column in CATEGORICAL_COLUMNS if column in columns]
        numeric = [column for column in columns if column not in categorical]
        return ColumnTransformer(
            [
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "encode",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=True,
                                    dtype=np.float32,
                                ),
                            ),
                        ]
                    ),
                    categorical,
                ),
                (
                    "num",
                    SimpleImputer(strategy="median"),
                    numeric,
                ),
            ],
            sparse_threshold=1.0,
        )

    def _make_estimator(self, n_estimators: int | None = None) -> XGBClassifier:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=int(n_estimators or self.n_estimators),
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            max_bin=self.max_bin,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y, eval_set=None):
        self.preprocessor_ = self._make_preprocessor(X.columns)
        transformed = self.preprocessor_.fit_transform(X, y)
        fit_kwargs = {"verbose": False}
        if eval_set is not None:
            X_eval, y_eval = eval_set
            fit_kwargs["eval_set"] = [
                (self.preprocessor_.transform(X_eval), np.asarray(y_eval))
            ]
        elif self.early_stopping_rounds is not None:
            raise ValueError("eval_set is required when early_stopping_rounds is set")
        self.model_ = self._make_estimator()
        self.model_.fit(transformed, np.asarray(y), **fit_kwargs)
        self.classes_ = np.asarray(self.model_.classes_)
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def refit_full(self, X, y, n_estimators: int):
        """Fit full data with the iteration count selected on validation."""
        self.preprocessor_ = self._make_preprocessor(X.columns)
        transformed = self.preprocessor_.fit_transform(X, y)
        self.model_ = self._make_estimator(n_estimators=n_estimators)
        self.model_.set_params(early_stopping_rounds=None)
        self.model_.fit(transformed, np.asarray(y), verbose=False)
        self.classes_ = np.asarray(self.model_.classes_)
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def best_iteration_count(self) -> int:
        check_is_fitted(self, ["model_"])
        best_iteration = getattr(self.model_, "best_iteration", None)
        return self.n_estimators if best_iteration is None else int(best_iteration) + 1

    def predict_proba(self, X):
        check_is_fitted(self, ["preprocessor_", "model_", "classes_"])
        transformed = self.preprocessor_.transform(X)
        prediction = np.asarray(self.model_.predict_proba(transformed), dtype=float)
        return np.clip(prediction, 0.0, 1.0)

    def feature_importance_frame(self):
        check_is_fitted(self, ["preprocessor_", "model_"])
        names = np.asarray(self.preprocessor_.get_feature_names_out(), dtype=object)
        importance = np.asarray(self.model_.feature_importances_, dtype=float)
        if len(names) != len(importance):
            raise ValueError(f"feature name/importance mismatch: {len(names)} != {len(importance)}")
        total = float(importance.sum())
        if total > 0.0:
            importance = importance / total
        cleaned = np.asarray([name.split("__", 1)[-1] for name in names], dtype=object)
        order = np.argsort(-importance, kind="stable")
        return cleaned[order], importance[order]

    @property
    def feature_importance_source(self) -> str:
        return "XGBoost gain"
