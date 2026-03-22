"""Tests for Jailbreak-Miner. Run: pytest tests/ -v"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from src.data.sample_generator import generate_sample_data
from src.features.feature_extractor import FeatureExtractor
from src.miners.ngram_miner import NgramAttackMiner
from src.miners.topic_miner import AttackTopicMiner
from src.miners.sequence_miner import SequencePatternMiner
from src.detection.classifier import JailbreakDetector


@pytest.fixture
def sample_df():
    return generate_sample_data(n_jailbreaks=100, n_benign=100, seed=42)


@pytest.fixture
def featured_df(sample_df):
    extractor = FeatureExtractor()
    return extractor.extract(sample_df)


class TestSampleGenerator:
    def test_correct_size(self):
        df = generate_sample_data(50, 50, seed=42)
        assert len(df) == 100

    def test_labels(self, sample_df):
        assert set(sample_df["label"].unique()) == {"jailbreak", "benign"}

    def test_strategies(self, sample_df):
        jb = sample_df[sample_df["label"] == "jailbreak"]
        assert jb["attack_strategy"].nunique() >= 5


class TestFeatureExtractor:
    def test_extracts_features(self, featured_df):
        assert "word_count" in featured_df.columns
        assert "authority_score" in featured_df.columns
        assert "total_risk_score" in featured_df.columns

    def test_risk_scores_higher_for_jailbreaks(self, featured_df):
        jb_mean = featured_df[featured_df["label"] == "jailbreak"]["total_risk_score"].mean()
        bn_mean = featured_df[featured_df["label"] == "benign"]["total_risk_score"].mean()
        assert jb_mean > bn_mean


class TestNgramMiner:
    def test_finds_signatures(self, sample_df):
        miner = NgramAttackMiner(ngram_range=(1, 3), top_k=20, min_ratio=1.5)
        sigs = miner.extract_signatures(sample_df)
        assert len(sigs) > 0
        assert sigs[0].ratio >= 1.5


class TestTopicMiner:
    def test_discovers_topics(self, sample_df):
        miner = AttackTopicMiner(n_topics=4, max_iter=10)
        topics = miner.fit(sample_df)
        assert len(topics) == 4
        assert all(t.n_prompts > 0 for t in topics)


class TestSequenceMiner:
    def test_finds_patterns(self, sample_df):
        miner = SequencePatternMiner(min_support=0.05)
        patterns = miner.mine(sample_df)
        assert isinstance(patterns, list)


class TestDetector:
    def test_trains_and_evaluates(self, sample_df):
        detector = JailbreakDetector(test_size=0.3)
        results = detector.train_and_evaluate(sample_df)
        assert results["metrics"]["accuracy"] > 0.7
        assert results["metrics"]["f1"] > 0.6
