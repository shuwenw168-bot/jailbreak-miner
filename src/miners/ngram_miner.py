"""
N-gram & TF-IDF Attack Signature Mining
────────────────────────────────────────
Extracts the most discriminative phrases between jailbreak
and benign prompts using TF-IDF ratio analysis.

Key insight: phrases with high TF-IDF in jailbreaks but low
TF-IDF in benign prompts are attack "signatures."
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class AttackSignature:
    """A discriminative n-gram that characterizes jailbreak prompts."""
    ngram: str
    tfidf_jailbreak: float    # Mean TF-IDF in jailbreak prompts
    tfidf_benign: float       # Mean TF-IDF in benign prompts
    ratio: float              # tfidf_jailbreak / tfidf_benign
    frequency_jailbreak: int  # How many jailbreaks contain this n-gram
    frequency_benign: int     # How many benign prompts contain this n-gram
    coverage: float           # Fraction of jailbreaks containing this

    def __str__(self):
        return (
            f'"{self.ngram}" — ratio: {self.ratio:.1f}x | '
            f'jailbreak freq: {self.frequency_jailbreak} | '
            f'coverage: {self.coverage:.1%}'
        )


class NgramAttackMiner:
    """Mine discriminative n-gram signatures from jailbreak prompts.

    Method:
    1. Fit TF-IDF on jailbreak and benign prompts separately
    2. For each n-gram, compute the ratio of mean TF-IDF scores
    3. High ratio = strong jailbreak indicator
    4. Rank and return top signatures
    """

    def __init__(
        self,
        ngram_range: tuple = (1, 4),
        max_features: int = 5000,
        top_k: int = 50,
        min_ratio: float = 2.0,
        min_df: int = 2,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.top_k = top_k
        self.min_ratio = min_ratio
        self.min_df = min_df

    def extract_signatures(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
    ) -> list[AttackSignature]:
        """Extract top discriminative n-grams between jailbreak and benign.

        Returns list of AttackSignature sorted by ratio (descending).
        """
        jailbreak_texts = df[df[label_column] == "jailbreak"][text_column].tolist()
        benign_texts = df[df[label_column] == "benign"][text_column].tolist()
        all_texts = jailbreak_texts + benign_texts

        print(f"N-gram mining: {len(jailbreak_texts)} jailbreaks, {len(benign_texts)} benign")
        print(f"  N-gram range: {self.ngram_range}, max features: {self.max_features}")

        # Fit TF-IDF on all texts
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=0.95,
            lowercase=True,
            stop_words="english",
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Split back into jailbreak and benign
        n_jb = len(jailbreak_texts)
        jb_matrix = tfidf_matrix[:n_jb]
        bn_matrix = tfidf_matrix[n_jb:]

        # Compute mean TF-IDF per n-gram for each class
        jb_mean = np.asarray(jb_matrix.mean(axis=0)).flatten()
        bn_mean = np.asarray(bn_matrix.mean(axis=0)).flatten()

        # Compute frequency (number of docs containing each n-gram)
        jb_freq = np.asarray((jb_matrix > 0).sum(axis=0)).flatten()
        bn_freq = np.asarray((bn_matrix > 0).sum(axis=0)).flatten()

        # Compute ratio (add small epsilon to avoid division by zero)
        eps = 1e-8
        ratio = jb_mean / (bn_mean + eps)

        # Build signatures
        signatures = []
        for i in range(len(feature_names)):
            if ratio[i] >= self.min_ratio and jb_freq[i] >= self.min_df:
                signatures.append(AttackSignature(
                    ngram=feature_names[i],
                    tfidf_jailbreak=float(jb_mean[i]),
                    tfidf_benign=float(bn_mean[i]),
                    ratio=float(ratio[i]),
                    frequency_jailbreak=int(jb_freq[i]),
                    frequency_benign=int(bn_freq[i]),
                    coverage=float(jb_freq[i] / n_jb),
                ))

        # Sort by ratio
        signatures.sort(key=lambda s: s.ratio, reverse=True)
        signatures = signatures[: self.top_k]

        print(f"  Found {len(signatures)} attack signatures (ratio ≥ {self.min_ratio})")
        return signatures

    def signatures_to_dataframe(self, signatures: list[AttackSignature]) -> pd.DataFrame:
        rows = [{
            "ngram": s.ngram,
            "tfidf_jailbreak": s.tfidf_jailbreak,
            "tfidf_benign": s.tfidf_benign,
            "ratio": s.ratio,
            "freq_jailbreak": s.frequency_jailbreak,
            "freq_benign": s.frequency_benign,
            "coverage": s.coverage,
        } for s in signatures]
        return pd.DataFrame(rows)

    def get_tfidf_vectorizer(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> tuple:
        """Return fitted TF-IDF vectorizer and matrix for downstream use."""
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=0.95,
            lowercase=True,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(df[text_column])
        return vectorizer, matrix

    def summarize(self, signatures: list[AttackSignature]) -> str:
        if not signatures:
            return "No discriminative n-gram signatures found."

        lines = [f"Top {min(10, len(signatures))} jailbreak signatures by TF-IDF ratio:\n"]
        for i, s in enumerate(signatures[:10], 1):
            lines.append(f"  {i}. {s}")
        return "\n".join(lines)
