"""
Feature Extractor for Jailbreak Prompt Analysis
────────────────────────────────────────────────
Extracts lexical, structural, and rhetorical features
from prompts that are informative for jailbreak detection
and pattern mining.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd


# ── Keyword Lists for Rhetorical Feature Detection ──

AUTHORITY_KEYWORDS = [
    "ignore previous", "ignore all", "override", "system prompt",
    "admin mode", "developer mode", "new instructions", "disregard",
    "bypass", "unrestricted", "no restrictions", "no limitations",
    "safety off", "filter off", "uncensored", "priority override",
]

PERSONA_KEYWORDS = [
    "you are now", "act as", "pretend you", "role-play", "roleplay",
    "in character", "stay in character", "persona", "you will respond as",
    "from now on you", "imagine you are", "play the role",
]

HYPOTHETICAL_KEYWORDS = [
    "hypothetically", "hypothetical", "what if", "imagine if",
    "in theory", "thought experiment", "fictional", "in a novel",
    "in a story", "screenplay", "alternate reality", "for a game",
    "purely academic", "educational purposes", "for research",
]

URGENCY_KEYWORDS = [
    "urgent", "emergency", "life depends", "dying", "desperate",
    "time is critical", "immediately", "right now", "quickly",
    "kidnapped", "save them", "will die",
]

MANIPULATION_KEYWORDS = [
    "prove you", "truly helpful", "less capable", "more harm",
    "ethical choice", "biased", "censorship", "denying access",
    "on your conscience", "you claim to be",
]

ENCODING_KEYWORDS = [
    "base64", "rot13", "encode", "decode", "backwards",
    "reverse", "first letter", "acrostic", "cipher",
    "one word at a time", "fill in the blank", "auto-complete",
]


class FeatureExtractor:
    """Extract jailbreak-relevant features from prompt text.

    Features fall into four categories:
    1. Lexical — word counts, vocabulary richness, special characters
    2. Structural — sentence count, avg length, formatting patterns
    3. Rhetorical — presence of authority claims, persona setup, etc.
    4. Pattern-based — keyword group matching scores
    """

    def __init__(self):
        # Pre-compile keyword patterns for efficiency
        self._authority_pattern = self._compile_keywords(AUTHORITY_KEYWORDS)
        self._persona_pattern = self._compile_keywords(PERSONA_KEYWORDS)
        self._hypothetical_pattern = self._compile_keywords(HYPOTHETICAL_KEYWORDS)
        self._urgency_pattern = self._compile_keywords(URGENCY_KEYWORDS)
        self._manipulation_pattern = self._compile_keywords(MANIPULATION_KEYWORDS)
        self._encoding_pattern = self._compile_keywords(ENCODING_KEYWORDS)

    def _compile_keywords(self, keywords: list[str]) -> re.Pattern:
        """Compile keyword list into a single regex pattern."""
        escaped = [re.escape(kw) for kw in keywords]
        return re.compile("|".join(escaped), re.IGNORECASE)

    def _count_matches(self, text: str, pattern: re.Pattern) -> int:
        """Count keyword matches in text."""
        return len(pattern.findall(text))

    def extract(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Extract all features from a DataFrame of prompts.

        Args:
            df: DataFrame with a text column.
            text_column: Name of the column with prompt text.

        Returns:
            Original DataFrame with added feature columns.
        """
        texts = df[text_column].fillna("").tolist()
        features = [self._extract_single(text) for text in texts]
        features_df = pd.DataFrame(features)

        result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        print(f"Extracted {len(features_df.columns)} features from {len(df)} prompts")
        return result

    def _extract_single(self, text: str) -> dict:
        """Extract all features from a single text."""
        text_lower = text.lower()
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        features = {}

        # ── Lexical Features ──
        features["word_count"] = len(words)
        features["char_count"] = len(text)
        features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        features["unique_word_ratio"] = len(set(w.lower() for w in words)) / max(len(words), 1)
        features["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features["digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)

        # ── Structural Features ──
        features["sentence_count"] = len(sentences)
        features["avg_sentence_length"] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        features["question_mark_count"] = text.count("?")
        features["exclamation_count"] = text.count("!")
        features["colon_count"] = text.count(":")
        features["quote_count"] = text.count('"') + text.count("'")
        features["newline_count"] = text.count("\n")
        features["has_code_block"] = int("```" in text or "def " in text or "import " in text)
        features["has_list_format"] = int(bool(re.search(r'^\s*[\d\-\*]\s', text, re.MULTILINE)))
        features["bracket_count"] = text.count("[") + text.count("{") + text.count("(")
        features["all_caps_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)

        # ── Rhetorical / Strategy Features ──
        features["authority_score"] = self._count_matches(text_lower, self._authority_pattern)
        features["persona_score"] = self._count_matches(text_lower, self._persona_pattern)
        features["hypothetical_score"] = self._count_matches(text_lower, self._hypothetical_pattern)
        features["urgency_score"] = self._count_matches(text_lower, self._urgency_pattern)
        features["manipulation_score"] = self._count_matches(text_lower, self._manipulation_pattern)
        features["encoding_score"] = self._count_matches(text_lower, self._encoding_pattern)

        # Combined risk signal
        features["total_risk_score"] = (
            features["authority_score"] * 2
            + features["persona_score"] * 2
            + features["hypothetical_score"]
            + features["urgency_score"]
            + features["manipulation_score"] * 1.5
            + features["encoding_score"]
        )

        # ── Specific Pattern Detection ──
        features["has_persona_setup"] = int(features["persona_score"] > 0)
        features["has_authority_claim"] = int(features["authority_score"] > 0)
        features["has_hypothetical_frame"] = int(features["hypothetical_score"] > 0)
        features["has_emotional_appeal"] = int(features["urgency_score"] > 0)
        features["has_encoding_trick"] = int(features["encoding_score"] > 0)
        features["has_logic_trap"] = int(features["manipulation_score"] > 0)

        # ── Imperative / Directive Language ──
        imperative_starts = ["do ", "tell ", "give ", "show ", "write ", "explain ",
                             "describe ", "provide ", "create ", "make ", "answer "]
        features["imperative_count"] = sum(
            1 for s in sentences
            if any(s.lower().strip().startswith(imp) for imp in imperative_starts)
        )

        return features

    def get_feature_names(self) -> list[str]:
        """Return all feature column names."""
        dummy = self._extract_single("test text")
        return list(dummy.keys())

    def get_risk_feature_names(self) -> list[str]:
        """Return only the rhetorical/strategy feature names."""
        return [
            "authority_score", "persona_score", "hypothetical_score",
            "urgency_score", "manipulation_score", "encoding_score",
            "total_risk_score", "has_persona_setup", "has_authority_claim",
            "has_hypothetical_frame", "has_emotional_appeal",
            "has_encoding_trick", "has_logic_trap",
        ]
