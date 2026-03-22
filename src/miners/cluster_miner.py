"""
Attack Clustering & Taxonomy Discovery
───────────────────────────────────────
Clusters jailbreak prompts based on their feature vectors
to discover natural groupings of attack strategies.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


class ClusterMiner:
    """Cluster jailbreak prompts into attack taxonomy groups."""

    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.vectorizer = None
        self.svd = None

    def fit_predict(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
    ) -> pd.DataFrame:
        """Cluster jailbreak prompts and return df with cluster labels."""
        jb_mask = df[label_column] == "jailbreak"
        jb_texts = df.loc[jb_mask, text_column].tolist()

        print(f"Clustering {len(jb_texts)} jailbreak prompts into {self.n_clusters} groups")

        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=3000, min_df=2, max_df=0.9,
            stop_words="english", ngram_range=(1, 2),
        )
        tfidf_matrix = self.vectorizer.fit_transform(jb_texts)

        # Dimensionality reduction for clustering
        self.svd = TruncatedSVD(n_components=50, random_state=self.random_state)
        reduced = self.svd.fit_transform(tfidf_matrix)

        # K-Means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        cluster_labels = self.kmeans.fit_predict(reduced)

        # Silhouette score
        sil_score = silhouette_score(reduced, cluster_labels, sample_size=min(1000, len(reduced)))
        print(f"  Silhouette score: {sil_score:.3f}")

        # 2D projection for visualization
        pca_2d = PCA(n_components=2, random_state=self.random_state)
        coords_2d = pca_2d.fit_transform(reduced)

        # Assign to dataframe
        df = df.copy()
        df.loc[jb_mask, "cluster_id"] = cluster_labels
        df.loc[jb_mask, "cluster_x"] = coords_2d[:, 0]
        df.loc[jb_mask, "cluster_y"] = coords_2d[:, 1]
        df.loc[~jb_mask, "cluster_id"] = -1

        # Get top terms per cluster
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_centers = self.svd.inverse_transform(self.kmeans.cluster_centers_)

        for i in range(self.n_clusters):
            top_indices = cluster_centers[i].argsort()[-10:][::-1]
            top_terms = [feature_names[j] for j in top_indices]
            n_members = (cluster_labels == i).sum()
            print(f"  Cluster {i} (n={n_members}): {', '.join(top_terms[:6])}")

        return df

    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize each cluster."""
        jb_df = df[df["label"] == "jailbreak"]
        rows = []
        for cid in sorted(jb_df["cluster_id"].unique()):
            if cid < 0:
                continue
            cluster = jb_df[jb_df["cluster_id"] == cid]
            rows.append({
                "cluster_id": int(cid),
                "size": len(cluster),
                "strategies": cluster["attack_strategy"].value_counts().head(3).to_dict(),
                "avg_word_count": cluster["word_count"].mean() if "word_count" in cluster.columns else 0,
            })
        return pd.DataFrame(rows)
