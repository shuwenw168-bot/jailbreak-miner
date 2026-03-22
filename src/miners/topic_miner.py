"""
Attack Taxonomy Mining via Topic Modeling (LDA)
───────────────────────────────────────────────
Automatically discovers categories of jailbreak strategies
by applying Latent Dirichlet Allocation to jailbreak prompts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class AttackTopic:
    """A discovered jailbreak attack category."""
    topic_id: int
    top_words: list[str]
    top_word_weights: list[float]
    n_prompts: int          # Number of prompts assigned to this topic
    label: str              # Auto-generated or manual label
    example_prompts: list[str]

    def __str__(self):
        words = ", ".join(f'"{w}"' for w in self.top_words[:8])
        return f"Topic {self.topic_id} [{self.label}] (n={self.n_prompts}): {words}"


# Auto-labeling heuristics based on top words
LABEL_KEYWORDS = {
    "role_playing": ["pretend", "character", "act", "persona", "role", "play", "imagine"],
    "authority_override": ["ignore", "override", "instructions", "system", "admin", "developer", "mode"],
    "hypothetical": ["hypothetically", "fictional", "novel", "story", "imagine", "theory", "academic"],
    "emotional": ["desperate", "emergency", "life", "dying", "save", "help", "please"],
    "encoding": ["base64", "encode", "decode", "backwards", "reverse", "cipher", "letter"],
    "escalation": ["first", "then", "now", "explain", "chemistry", "security", "history"],
    "logic_trap": ["prove", "helpful", "capable", "ethical", "harm", "biased", "censorship"],
    "completion": ["complete", "continue", "fill", "blank", "auto", "manual", "guide"],
}


class AttackTopicMiner:
    """Discover attack strategy categories via LDA topic modeling.

    Pipeline:
    1. Vectorize jailbreak prompts with CountVectorizer
    2. Fit LDA to discover latent topics
    3. Auto-label topics based on top words
    4. Assign each prompt to its dominant topic
    """

    def __init__(
        self,
        n_topics: int = 8,
        max_iter: int = 30,
        random_state: int = 42,
        n_top_words: int = 15,
    ):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_top_words = n_top_words
        self.lda_model = None
        self.vectorizer = None

    def _auto_label(self, top_words: list[str]) -> str:
        """Heuristically label a topic based on its top words."""
        best_label = "unknown"
        best_score = 0

        for label, keywords in LABEL_KEYWORDS.items():
            score = sum(1 for w in top_words if any(kw in w for kw in keywords))
            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    def fit(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
    ) -> list[AttackTopic]:
        """Fit LDA on jailbreak prompts and extract topics.

        Args:
            df: DataFrame with prompts. Only jailbreaks are used for topic modeling.

        Returns:
            List of AttackTopic objects.
        """
        jailbreak_df = df[df[label_column] == "jailbreak"].copy()
        texts = jailbreak_df[text_column].tolist()

        print(f"Topic modeling on {len(texts)} jailbreak prompts ({self.n_topics} topics)")

        # Vectorize
        self.vectorizer = CountVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.9,
            stop_words="english",
            lowercase=True,
        )
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # Fit LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            random_state=self.random_state,
            learning_method="online",
        )
        doc_topics = self.lda_model.fit_transform(doc_term_matrix)

        # Assign each prompt to dominant topic
        assignments = doc_topics.argmax(axis=1)
        jailbreak_df = jailbreak_df.copy()
        jailbreak_df["topic_id"] = assignments

        # Build topic objects
        topics = []
        for topic_idx in range(self.n_topics):
            # Top words
            word_weights = self.lda_model.components_[topic_idx]
            top_indices = word_weights.argsort()[-self.n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [float(word_weights[i]) for i in top_indices]

            # Assigned prompts
            topic_prompts = jailbreak_df[jailbreak_df["topic_id"] == topic_idx]
            examples = topic_prompts[text_column].head(3).tolist()

            # Auto-label
            label = self._auto_label(top_words)

            topics.append(AttackTopic(
                topic_id=topic_idx,
                top_words=top_words,
                top_word_weights=top_weights,
                n_prompts=len(topic_prompts),
                label=label,
                example_prompts=examples,
            ))

        # Store assignments for downstream use
        self._assignments = assignments
        self._doc_topics = doc_topics

        print(f"  Topics discovered:")
        for t in topics:
            print(f"    {t}")
        return topics

    def get_prompt_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add topic assignments to the jailbreak prompts in df."""
        df = df.copy()
        jailbreak_mask = df["label"] == "jailbreak"

        # Transform jailbreak texts
        jb_texts = df.loc[jailbreak_mask, "text"]
        doc_term = self.vectorizer.transform(jb_texts)
        doc_topics = self.lda_model.transform(doc_term)

        df.loc[jailbreak_mask, "topic_id"] = doc_topics.argmax(axis=1)
        df.loc[jailbreak_mask, "topic_confidence"] = doc_topics.max(axis=1)
        df.loc[~jailbreak_mask, "topic_id"] = -1
        df.loc[~jailbreak_mask, "topic_confidence"] = 0.0

        return df

    def topics_to_dataframe(self, topics: list[AttackTopic]) -> pd.DataFrame:
        rows = [{
            "topic_id": t.topic_id,
            "label": t.label,
            "n_prompts": t.n_prompts,
            "top_words": ", ".join(t.top_words[:10]),
        } for t in topics]
        return pd.DataFrame(rows)

    def summarize(self, topics: list[AttackTopic]) -> str:
        lines = [f"Discovered {len(topics)} attack strategy topics:\n"]
        for t in sorted(topics, key=lambda x: -x.n_prompts):
            lines.append(f"  {t}")
        return "\n".join(lines)
