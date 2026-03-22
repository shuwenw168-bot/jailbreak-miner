"""
Jailbreak Attack Visualizations
────────────────────────────────
Publication-ready figures for jailbreak pattern analysis.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
    "figure.dpi": 150,
})


class AttackPlotter:
    def __init__(self, output_dir: str = "results/figures", dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def _save(self, fig, name):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path}")

    def plot_ngram_signatures(self, sig_df: pd.DataFrame, top_n: int = 20):
        """Bar chart of top attack n-gram signatures."""
        if len(sig_df) == 0:
            return
        top = sig_df.head(top_n).sort_values("ratio")
        fig, ax = plt.subplots(figsize=(11, max(4, len(top) * 0.4)))
        colors = ["#e74c3c" if r > 10 else "#f39c12" if r > 5 else "#3498db" for r in top["ratio"]]
        ax.barh(top["ngram"], top["ratio"], color=colors, edgecolor="white")
        ax.set_xlabel("TF-IDF Ratio (Jailbreak / Benign)")
        ax.set_title("Top Jailbreak Attack Signatures by TF-IDF Ratio")
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
        self._save(fig, "ngram_signatures")

    def plot_strategy_distribution(self, df: pd.DataFrame):
        """Pie chart of attack strategy distribution."""
        jb = df[df["label"] == "jailbreak"]
        counts = jb["attack_strategy"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = sns.color_palette("Set2", len(counts))
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, pctdistance=0.85,
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title("Distribution of Jailbreak Attack Strategies")
        self._save(fig, "strategy_distribution")

    def plot_feature_comparison(self, df: pd.DataFrame):
        """Box plots comparing key features between jailbreak and benign."""
        features = [
            "word_count", "authority_score", "persona_score",
            "hypothetical_score", "total_risk_score",
        ]
        features = [f for f in features if f in df.columns]

        fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 5))
        if len(features) == 1:
            axes = [axes]

        for ax, feat in zip(axes, features):
            sns.boxplot(data=df, x="label", y=feat, ax=ax, palette={"jailbreak": "#e74c3c", "benign": "#2ecc71"})
            ax.set_title(feat.replace("_", " ").title())
            ax.set_xlabel("")

        plt.suptitle("Feature Distribution: Jailbreak vs Benign", y=1.02, fontsize=14)
        plt.tight_layout()
        self._save(fig, "feature_comparison")

    def plot_topic_distribution(self, topics_df: pd.DataFrame):
        """Bar chart of topic sizes."""
        if len(topics_df) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        topics_df = topics_df.sort_values("n_prompts", ascending=True)
        colors = sns.color_palette("Set2", len(topics_df))
        ax.barh(topics_df["label"] + " (T" + topics_df["topic_id"].astype(str) + ")",
                topics_df["n_prompts"], color=colors, edgecolor="white")
        ax.set_xlabel("Number of Prompts")
        ax.set_title("Attack Strategy Topics (LDA)")
        self._save(fig, "topic_distribution")

    def plot_sequence_patterns(self, patterns_df: pd.DataFrame, top_n: int = 12):
        """Horizontal bar chart of sequential patterns."""
        if len(patterns_df) == 0:
            return
        top = patterns_df.head(top_n).sort_values("support")
        fig, ax = plt.subplots(figsize=(12, max(4, len(top) * 0.45)))
        colors = ["#e74c3c" if s > 0.2 else "#f39c12" if s > 0.1 else "#3498db" for s in top["support"]]
        ax.barh(top["pattern"], top["support"], color=colors, edgecolor="white")
        ax.set_xlabel("Support (fraction of jailbreaks)")
        ax.set_title("Frequent Sequential Attack Patterns")
        self._save(fig, "sequence_patterns")

    def plot_cluster_map(self, df: pd.DataFrame):
        """2D scatter plot of jailbreak clusters."""
        jb = df[(df["label"] == "jailbreak") & (df.get("cluster_id", pd.Series([-1])) >= 0)]
        if "cluster_x" not in jb.columns or len(jb) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            jb["cluster_x"], jb["cluster_y"],
            c=jb["cluster_id"], cmap="Set2", alpha=0.6, s=30, edgecolors="white", linewidth=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster ID")
        ax.set_title("Jailbreak Prompt Clusters (PCA Projection)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        self._save(fig, "cluster_map")

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Jailbreak"],
            yticklabels=["Benign", "Jailbreak"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Jailbreak Detector: Confusion Matrix")
        self._save(fig, "confusion_matrix")

    def plot_top_features(self, features_df: pd.DataFrame, top_n: int = 20):
        """Top classifier features."""
        if len(features_df) == 0:
            return

        jb_feats = features_df[features_df["direction"] == "jailbreak"].head(top_n)
        if len(jb_feats) == 0:
            return

        jb_feats = jb_feats.sort_values("weight")
        fig, ax = plt.subplots(figsize=(10, max(4, len(jb_feats) * 0.35)))
        ax.barh(jb_feats["feature"], jb_feats["weight"], color="#e74c3c", edgecolor="white")
        ax.set_xlabel("Classifier Weight")
        ax.set_title("Top Features Indicating Jailbreak")
        self._save(fig, "top_classifier_features")

    def generate_all(self, df, sig_df=None, topics_df=None, patterns_df=None,
                     detection_results=None):
        """Generate all available visualizations."""
        print("\nGenerating visualizations...")
        self.plot_strategy_distribution(df)
        if "word_count" in df.columns:
            self.plot_feature_comparison(df)
        if sig_df is not None and len(sig_df) > 0:
            self.plot_ngram_signatures(sig_df)
        if topics_df is not None and len(topics_df) > 0:
            self.plot_topic_distribution(topics_df)
        if patterns_df is not None and len(patterns_df) > 0:
            self.plot_sequence_patterns(patterns_df)
        if "cluster_x" in df.columns:
            self.plot_cluster_map(df)
        if detection_results:
            self.plot_confusion_matrix(detection_results["confusion_matrix"])
            if len(detection_results.get("top_features", [])) > 0:
                self.plot_top_features(detection_results["top_features"])
        print(f"All figures saved to {self.output_dir}/")
