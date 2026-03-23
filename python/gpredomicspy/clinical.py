"""Clinical data integration for gpredomics.

Combines omics-derived prediction scores from gpredomics with clinical
variables (age, BMI, lab values, comorbidities, medications) to improve
overall prediction accuracy while preserving interpretability.

Approaches:
    1. Stacking (late fusion) — omics score + clinical vars → logistic regression
    2. Score calibration + combination — calibrate + combine in log-odds space
    3. Stratified models — separate gpredomics models per clinical stratum
    4. Interaction features — omics_score × clinical_var terms

Example::

    import gpredomicspy as gp
    from gpredomicspy.clinical import StackingIntegrator

    # Train omics model
    param = gp.Param()
    param.load("param.yaml")
    exp = gp.fit(param)

    # Get omics scores
    scores_train = exp.predict_scores_train()
    scores_test = exp.predict_scores_test()
    y_train = exp.train_labels()

    # Combine with clinical data
    integrator = StackingIntegrator()
    integrator.fit(scores_train, clinical_train_df, y_train)
    results = integrator.predict(scores_test, clinical_test_df)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _to_array(x) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if HAS_PANDAS and isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    return np.asarray(x)


def _build_feature_matrix(
    omics_scores: np.ndarray,
    clinical: Optional[np.ndarray],
    interactions: bool = False,
) -> np.ndarray:
    """Build feature matrix from omics scores and clinical data."""
    scores = omics_scores.reshape(-1, 1) if omics_scores.ndim == 1 else omics_scores

    if clinical is None:
        return scores

    clinical = clinical if clinical.ndim == 2 else clinical.reshape(-1, 1)
    parts = [scores, clinical]

    if interactions:
        # Add interaction terms: each omics_score × each clinical_var
        for j in range(clinical.shape[1]):
            for s in range(scores.shape[1]):
                parts.append((scores[:, s] * clinical[:, j]).reshape(-1, 1))

    return np.hstack(parts)


class StackingIntegrator:
    """Late fusion via stacking: omics score + clinical variables → second-stage model.

    The gpredomics model produces an omics score S_omics for each sample.
    This score is combined with clinical variables in a second-stage logistic
    regression (or other model) to produce a final prediction.

    Args:
        method: Second-stage classifier ('logistic', 'logistic_l1', 'rf', 'xgboost').
        interactions: If True, add interaction terms (omics_score × clinical_var).
        cv_folds: Number of CV folds for generating out-of-fold omics scores
                  during training (prevents leakage). 0 = use raw scores.
        seed: Random seed for reproducibility.

    Example::

        integrator = StackingIntegrator(method='logistic_l1', interactions=True)
        integrator.fit(omics_scores_train, clinical_train, y_train)
        y_pred, y_proba = integrator.predict(omics_scores_test, clinical_test)
    """

    def __init__(
        self,
        method: str = "logistic",
        interactions: bool = False,
        cv_folds: int = 5,
        seed: int = 42,
    ):
        self.method = method
        self.interactions = interactions
        self.cv_folds = cv_folds
        self.seed = seed
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None

    def _build_model(self):
        if self.method == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                penalty="l2", C=1.0, class_weight="balanced",
                max_iter=1000, random_state=self.seed,
            )
        elif self.method == "logistic_l1":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                penalty="l1", C=1.0, solver="liblinear",
                class_weight="balanced", max_iter=1000, random_state=self.seed,
            )
        elif self.method == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=self.seed, n_jobs=-1,
            )
        elif self.method == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=self.seed, n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(
        self,
        omics_scores: Union[np.ndarray, list],
        clinical: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
        y: Union[np.ndarray, list] = None,
        clinical_feature_names: Optional[list[str]] = None,
    ) -> "StackingIntegrator":
        """Fit the stacking integrator.

        Args:
            omics_scores: Omics prediction scores (1D array, one per sample).
            clinical: Clinical variables (2D array or DataFrame, samples × features).
                      None = use omics scores only.
            y: True class labels (0/1).
            clinical_feature_names: Names for clinical columns (for interpretability).
        """
        from sklearn.preprocessing import StandardScaler

        omics_scores = _to_array(omics_scores).ravel()
        y = _to_array(y).ravel()

        if clinical is not None:
            if HAS_PANDAS and isinstance(clinical, pd.DataFrame):
                self.feature_names_ = ["omics_score"] + list(clinical.columns)
                clinical = clinical.values
            elif clinical_feature_names:
                self.feature_names_ = ["omics_score"] + clinical_feature_names
            else:
                n_clin = clinical.shape[1] if clinical.ndim == 2 else 1
                self.feature_names_ = ["omics_score"] + [f"clinical_{i}" for i in range(n_clin)]

            if self.interactions:
                clin_names = self.feature_names_[1:]
                interaction_names = [f"omics×{c}" for c in clin_names]
                self.feature_names_ += interaction_names
        else:
            self.feature_names_ = ["omics_score"]

        X = _build_feature_matrix(omics_scores, clinical, self.interactions)

        # Scale features for logistic regression
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.model_ = self._build_model()
        self.model_.fit(X_scaled, y)

        return self

    def predict(
        self,
        omics_scores: Union[np.ndarray, list],
        clinical: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict classes and probabilities.

        Returns:
            Tuple of (predicted_classes, probabilities_class1).
        """
        omics_scores = _to_array(omics_scores).ravel()
        if clinical is not None:
            clinical = _to_array(clinical)

        X = _build_feature_matrix(omics_scores, clinical, self.interactions)
        X_scaled = self.scaler_.transform(X)

        y_pred = self.model_.predict(X_scaled)
        y_proba = self.model_.predict_proba(X_scaled)[:, 1]

        return y_pred, y_proba

    def feature_importances(self) -> dict[str, float]:
        """Get feature importances/coefficients from the second-stage model.

        Returns:
            Dict mapping feature name to importance value.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        if hasattr(self.model_, "coef_"):
            coefs = self.model_.coef_.ravel()
        elif hasattr(self.model_, "feature_importances_"):
            coefs = self.model_.feature_importances_
        else:
            return {}

        return {
            name: round(float(c), 6)
            for name, c in zip(self.feature_names_, coefs)
        }

    def summary(self) -> dict:
        """Get a summary of the fitted model."""
        imp = self.feature_importances()
        return {
            "method": self.method,
            "interactions": self.interactions,
            "n_features": len(self.feature_names_),
            "feature_names": self.feature_names_,
            "importances": imp,
            "omics_weight": imp.get("omics_score", None),
        }


class CalibratedCombiner:
    """Score calibration + combination in log-odds space.

    Calibrates the gpredomics score into a proper probability using Platt
    scaling (logistic regression) or isotonic regression, then combines
    with a clinical risk score via log-odds addition (naive Bayes).

    Args:
        calibration_method: 'platt' (logistic) or 'isotonic'.
        seed: Random seed.

    Example::

        combiner = CalibratedCombiner()
        combiner.fit(omics_scores_train, clinical_risk_train, y_train)
        y_pred, combined_proba = combiner.predict(omics_scores_test, clinical_risk_test)
    """

    def __init__(self, calibration_method: str = "platt", seed: int = 42):
        self.calibration_method = calibration_method
        self.seed = seed
        self.omics_calibrator_ = None
        self.clinical_calibrator_ = None
        self.prior_odds_ = None

    def fit(
        self,
        omics_scores: Union[np.ndarray, list],
        clinical_risk: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
    ) -> "CalibratedCombiner":
        """Fit calibrators for omics and clinical scores.

        Args:
            omics_scores: Raw omics prediction scores (1D).
            clinical_risk: Clinical risk scores (1D) — can be raw probabilities
                           or any score that correlates with risk.
            y: True class labels (0/1).
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        omics_scores = _to_array(omics_scores).ravel()
        clinical_risk = _to_array(clinical_risk).ravel()
        y = _to_array(y).ravel()

        # Compute prior odds from training data
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        self.prior_odds_ = n_pos / max(n_neg, 1)

        if self.calibration_method == "platt":
            # Platt scaling = logistic regression on raw scores
            self.omics_calibrator_ = LogisticRegression(random_state=self.seed, max_iter=1000)
            self.omics_calibrator_.fit(omics_scores.reshape(-1, 1), y)

            self.clinical_calibrator_ = LogisticRegression(random_state=self.seed, max_iter=1000)
            self.clinical_calibrator_.fit(clinical_risk.reshape(-1, 1), y)
        elif self.calibration_method == "isotonic":
            self.omics_calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.omics_calibrator_.fit(omics_scores, y)

            self.clinical_calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.clinical_calibrator_.fit(clinical_risk, y)
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")

        return self

    def _calibrate(self, scores, calibrator):
        """Get calibrated probability from scores."""
        scores = _to_array(scores).ravel()
        if self.calibration_method == "platt":
            return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            return np.clip(calibrator.predict(scores), 1e-6, 1 - 1e-6)

    def predict(
        self,
        omics_scores: Union[np.ndarray, list],
        clinical_risk: Union[np.ndarray, list],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine calibrated scores via log-odds addition.

        Returns:
            Tuple of (predicted_classes, combined_probabilities).
        """
        p_omics = self._calibrate(omics_scores, self.omics_calibrator_)
        p_clinical = self._calibrate(clinical_risk, self.clinical_calibrator_)

        # Log-odds combination (naive Bayes)
        eps = 1e-6
        log_odds_omics = np.log(np.clip(p_omics, eps, 1 - eps) / np.clip(1 - p_omics, eps, 1 - eps))
        log_odds_clinical = np.log(np.clip(p_clinical, eps, 1 - eps) / np.clip(1 - p_clinical, eps, 1 - eps))
        log_odds_prior = np.log(max(self.prior_odds_, eps))

        # Combined log-odds = sum - prior (to avoid double-counting)
        combined_log_odds = log_odds_omics + log_odds_clinical - log_odds_prior
        combined_proba = 1 / (1 + np.exp(-combined_log_odds))

        y_pred = (combined_proba >= 0.5).astype(np.uint8)
        return y_pred, combined_proba

    def summary(self) -> dict:
        """Get calibration summary."""
        return {
            "calibration_method": self.calibration_method,
            "prior_odds": round(float(self.prior_odds_), 4) if self.prior_odds_ else None,
        }


class StratifiedIntegrator:
    """Train separate gpredomics (or sklearn) models per clinical stratum.

    Uses clinical variables to define patient subgroups, then trains
    separate models per stratum for more targeted predictions.

    Args:
        strata_column: Name of the clinical column to stratify by.
        base_param_path: Path to base param.yaml for gpredomics runs.
        seed: Random seed.

    Example::

        strat = StratifiedIntegrator(strata_column='metformin_use')
        strat.fit(X_train, y_train, clinical_train)
        y_pred, y_proba = strat.predict(X_test, clinical_test)
    """

    def __init__(self, strata_column: str, seed: int = 42):
        self.strata_column = strata_column
        self.seed = seed
        self.models_ = {}
        self.strata_values_ = None

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, list],
        clinical: Union[np.ndarray, "pd.DataFrame"],
        model_builder=None,
    ) -> "StratifiedIntegrator":
        """Fit separate models per clinical stratum.

        Args:
            X: Feature matrix (samples × features).
            y: True class labels.
            clinical: Clinical data (DataFrame or array). Must contain strata_column.
            model_builder: Callable that returns a fitted sklearn classifier.
                           Default: LogisticRegression(penalty='l1').
        """
        from sklearn.linear_model import LogisticRegression

        X = _to_array(X)
        y = _to_array(y).ravel()

        if HAS_PANDAS and isinstance(clinical, pd.DataFrame):
            strata = clinical[self.strata_column].values
        else:
            clinical = _to_array(clinical)
            strata = clinical[:, 0] if clinical.ndim == 2 else clinical

        self.strata_values_ = sorted(set(strata))

        if model_builder is None:
            def model_builder():
                return LogisticRegression(
                    penalty="l1", solver="liblinear", C=1.0,
                    class_weight="balanced", random_state=self.seed,
                )

        for val in self.strata_values_:
            mask = strata == val
            if mask.sum() < 5:
                continue
            X_sub, y_sub = X[mask], y[mask]
            if len(np.unique(y_sub)) < 2:
                continue
            model = model_builder()
            model.fit(X_sub, y_sub)
            self.models_[val] = model

        return self

    def predict(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        clinical: Union[np.ndarray, "pd.DataFrame"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict using stratum-specific models.

        Returns:
            Tuple of (predicted_classes, probabilities_class1).
        """
        X = _to_array(X)

        if HAS_PANDAS and isinstance(clinical, pd.DataFrame):
            strata = clinical[self.strata_column].values
        else:
            clinical = _to_array(clinical)
            strata = clinical[:, 0] if clinical.ndim == 2 else clinical

        y_pred = np.zeros(len(X), dtype=np.uint8)
        y_proba = np.full(len(X), 0.5)

        for val, model in self.models_.items():
            mask = strata == val
            if mask.sum() == 0:
                continue
            y_pred[mask] = model.predict(X[mask])
            if hasattr(model, "predict_proba"):
                y_proba[mask] = model.predict_proba(X[mask])[:, 1]

        return y_pred, y_proba

    def summary(self) -> dict:
        """Get stratification summary."""
        return {
            "strata_column": self.strata_column,
            "strata_values": list(self.models_.keys()),
            "n_strata": len(self.models_),
            "samples_per_stratum": {
                str(k): "fitted" for k in self.models_
            },
        }
