"""
Jailbreak Detection Classifier
───────────────────────────────
Lightweight TF-IDF + ML classifier for detecting jailbreak prompts.
Serves as a practical baseline that works without GPU or large models.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline


class JailbreakDetector:
    """Detect jailbreak prompts using TF-IDF features + classifier.

    Supports multiple classifiers as baselines:
    - Logistic Regression (default, fast, interpretable)
    - Random Forest
    - Gradient Boosting
    """

    CLASSIFIERS = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    def __init__(
        self,
        classifier: str = "logistic_regression",
        ngram_range: tuple = (1, 3),
        max_features: int = 5000,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.classifier_name = classifier
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = None
        self.results = None

    def _build_pipeline(self) -> Pipeline:
        clf_fn = self.CLASSIFIERS.get(self.classifier_name)
        if clf_fn is None:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")

        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=2,
                max_df=0.95,
                stop_words="english",
                sublinear_tf=True,
            )),
            ("clf", clf_fn()),
        ])

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
        feature_df: pd.DataFrame = None,
    ) -> dict:
        """Train classifier and evaluate with train/test split + cross-validation.

        Args:
            df: DataFrame with text and labels.
            feature_df: Optional pre-extracted features to concatenate with TF-IDF.

        Returns:
            Dict with metrics, confusion matrix, and feature importances.
        """
        texts = df[text_column].tolist()
        labels = (df[label_column] == "jailbreak").astype(int).tolist()

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        print(f"Training {self.classifier_name} detector")
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

        # Build and train pipeline
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # Predict
        y_pred = self.pipeline.predict(X_test)
        y_prob = None
        if hasattr(self.pipeline, "predict_proba"):
            y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # Cross-validation
        cv_scores = cross_val_score(
            self._build_pipeline(), texts, labels, cv=5, scoring="f1"
        )
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Feature importances (for logistic regression)
        top_features = self._get_top_features()

        self.results = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": classification_report(
                y_test, y_pred, target_names=["benign", "jailbreak"]
            ),
            "top_features": top_features,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"\n  Results:")
        print(f"    Accuracy:  {metrics['accuracy']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1:        {metrics['f1']:.3f}")
        print(f"    CV F1:     {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")

        return self.results

    def _get_top_features(self, top_n: int = 30) -> pd.DataFrame:
        """Extract most important features from the trained classifier."""
        if self.pipeline is None:
            return pd.DataFrame()

        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        feature_names = tfidf.get_feature_names_out()

        if hasattr(clf, "coef_"):
            # Logistic regression — coefficients
            coefs = clf.coef_.flatten()
            top_jailbreak_idx = coefs.argsort()[-top_n:][::-1]
            top_benign_idx = coefs.argsort()[:top_n]

            rows = []
            for idx in top_jailbreak_idx:
                rows.append({
                    "feature": feature_names[idx],
                    "weight": coefs[idx],
                    "direction": "jailbreak",
                })
            for idx in top_benign_idx:
                rows.append({
                    "feature": feature_names[idx],
                    "weight": coefs[idx],
                    "direction": "benign",
                })
            return pd.DataFrame(rows)

        elif hasattr(clf, "feature_importances_"):
            # Tree-based — feature importances
            importances = clf.feature_importances_
            top_idx = importances.argsort()[-top_n:][::-1]
            rows = [{
                "feature": feature_names[idx],
                "weight": importances[idx],
                "direction": "importance",
            } for idx in top_idx]
            return pd.DataFrame(rows)

        return pd.DataFrame()

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict jailbreak probability for new texts."""
        if self.pipeline is None:
            raise ValueError("Detector not trained. Call train_and_evaluate() first.")
        return self.pipeline.predict_proba(texts)[:, 1]

    def summarize(self) -> str:
        if self.results is None:
            return "Detector not yet trained."

        m = self.results["metrics"]
        lines = [
            f"Jailbreak Detector ({self.classifier_name}):",
            f"  Accuracy:  {m['accuracy']:.3f}",
            f"  Precision: {m['precision']:.3f}",
            f"  Recall:    {m['recall']:.3f}",
            f"  F1 Score:  {m['f1']:.3f}",
            f"  CV F1:     {m['cv_f1_mean']:.3f} ± {m['cv_f1_std']:.3f}",
            f"\n{self.results['classification_report']}",
        ]
        return "\n".join(lines)
