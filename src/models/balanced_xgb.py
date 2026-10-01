"""XGBoost classifier that balances classes within each fit (including CV folds)."""

from __future__ import annotations

import sys

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Small scikit-learn-compatible wrapper for weighted multiclass XGBoost."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_child_weight: float = 2.0,
        gamma: float = 0.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        class_weight_strength: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.class_weight_strength = class_weight_strength
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "XGBoost is missing from the Python interpreter running this script. "
                f"Install it with: \"{sys.executable}\" -m pip install xgboost==3.4.1"
            ) from error

        self.label_encoder_ = LabelEncoder().fit(y)
        encoded_y = self.label_encoder_.transform(y)
        balanced_weights = compute_sample_weight(class_weight="balanced", y=encoded_y)
        weights = 1.0 + self.class_weight_strength * (balanced_weights - 1.0)
        self.estimator_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        self.estimator_.fit(X, encoded_y, sample_weight=weights)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        encoded = self.estimator_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
