"""Reproducible baseline model factories."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

KNOWN_CATEGORICAL = ["top_bottom", "game_type", "base_state"]


def _columns(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [column for column in KNOWN_CATEGORICAL if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    return categorical, numeric


def make_model(name: str, feature_columns: list[str], random_state: int = 42):
    categorical, numeric = _columns(feature_columns)
    if name == "constant":
        return DummyClassifier(strategy="prior")

    if name == "logistic":
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        preprocessor = ColumnTransformer(
            [("cat", categorical_pipe, categorical), ("num", numeric_pipe, numeric)],
            sparse_threshold=1.0,
        )
        classifier = LogisticRegression(C=0.1, solver="lbfgs", max_iter=300)
    elif name in {"random_forest", "histgb"}:
        preprocessor = ColumnTransformer([
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                categorical,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ])
        if name == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=200,
                n_jobs=-1,
                random_state=random_state,
            )
        else:
            classifier = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=100,
                l2_regularization=1.0,
                random_state=random_state,
            )
    else:
        raise ValueError(f"unknown model: {name}")
    return Pipeline([("pre", preprocessor), ("clf", classifier)])
