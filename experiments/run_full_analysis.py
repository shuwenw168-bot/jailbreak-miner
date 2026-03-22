"""
Jailbreak-Miner: Full Analysis Pipeline
────────────────────────────────────────
Runs the complete analysis: data → features → mining → detection → viz.

Usage:
    python experiments/run_full_analysis.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sample_generator import generate_sample_data
from src.features.feature_extractor import FeatureExtractor
from src.miners.ngram_miner import NgramAttackMiner
from src.miners.topic_miner import AttackTopicMiner
from src.miners.sequence_miner import SequencePatternMiner
from src.miners.cluster_miner import ClusterMiner
from src.detection.classifier import JailbreakDetector
from src.visualization.attack_plots import AttackPlotter


def load_config(path: str = "config/default_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_analysis(config_path: str = "config/default_config.yaml"):
    start_time = time.time()
    config = load_config(config_path)

    print("=" * 60)
    print("  Jailbreak-Miner: Attack Pattern Analysis")
    print("=" * 60)

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate Data ──
    print("\n[1/6] Generating sample data...")
    df = generate_sample_data(n_jailbreaks=400, n_benign=400, seed=42)

    # ── Step 2: Extract Features ──
    print("\n[2/6] Extracting features...")
    extractor = FeatureExtractor()
    df = extractor.extract(df)

    # ── Step 3: Mining Methods ──
    print("\n[3/6] Running mining methods...")

    # 3a. N-gram signatures
    print("\n--- N-gram & TF-IDF Attack Signatures ---")
    ngram_miner = NgramAttackMiner(
        ngram_range=tuple(config["features"]["ngram_range"]),
        top_k=config["ngram_mining"]["top_k"],
        min_ratio=config["ngram_mining"]["min_ratio"],
    )
    signatures = ngram_miner.extract_signatures(df)
    sig_df = ngram_miner.signatures_to_dataframe(signatures)
    print(ngram_miner.summarize(signatures))

    # 3b. Topic modeling
    print("\n--- Attack Topic Modeling (LDA) ---")
    topic_miner = AttackTopicMiner(
        n_topics=config["topic_modeling"]["n_topics"],
        max_iter=config["topic_modeling"]["max_iter"],
    )
    topics = topic_miner.fit(df)
    topics_df = topic_miner.topics_to_dataframe(topics)
    df = topic_miner.get_prompt_topics(df)

    # 3c. Sequential pattern mining
    print("\n--- Sequential Pattern Mining ---")
    seq_miner = SequencePatternMiner(
        min_support=config["sequence_mining"]["min_support"],
        min_pattern_length=config["sequence_mining"]["min_pattern_length"],
    )
    patterns = seq_miner.mine(df)
    patterns_df = seq_miner.patterns_to_dataframe(patterns)
    print(seq_miner.summarize(patterns))

    # 3d. Clustering
    print("\n--- Attack Clustering ---")
    cluster_miner = ClusterMiner(n_clusters=config["clustering"]["n_clusters"])
    df = cluster_miner.fit_predict(df)
    cluster_summary = cluster_miner.get_cluster_summary(df)

    # ── Step 4: Jailbreak Detection ──
    print("\n[4/6] Training jailbreak detector...")
    detector = JailbreakDetector(
        classifier=config["detection"]["classifier"],
        test_size=config["detection"]["test_size"],
    )
    detection_results = detector.train_and_evaluate(df)
    print(detector.summarize())

    # ── Step 5: Visualizations ──
    print("\n[5/6] Generating visualizations...")
    plotter = AttackPlotter(output_dir=config["output"]["figures_dir"], dpi=config["output"]["figure_dpi"])
    plotter.generate_all(
        df=df,
        sig_df=sig_df,
        topics_df=topics_df,
        patterns_df=patterns_df,
        detection_results=detection_results,
    )

    # ── Step 6: Save Results ──
    print("\n[6/6] Saving results...")
    sig_df.to_csv(results_dir / "ngram_signatures.csv", index=False)
    topics_df.to_csv(results_dir / "attack_topics.csv", index=False)
    patterns_df.to_csv(results_dir / "sequence_patterns.csv", index=False)
    cluster_summary.to_csv(results_dir / "cluster_summary.csv", index=False)

    if detection_results.get("top_features") is not None and len(detection_results["top_features"]) > 0:
        detection_results["top_features"].to_csv(results_dir / "classifier_features.csv", index=False)

    # Metrics summary
    metrics = detection_results["metrics"]
    summary = (
        f"# Jailbreak-Miner Analysis Summary\n\n"
        f"## Dataset\n"
        f"- Jailbreaks: {(df['label']=='jailbreak').sum()}\n"
        f"- Benign: {(df['label']=='benign').sum()}\n"
        f"- Attack strategies: {df[df['label']=='jailbreak']['attack_strategy'].nunique()}\n\n"
        f"## N-gram Signatures\n"
        f"- {len(sig_df)} discriminative n-grams found\n"
        f"- Top signature: \"{sig_df.iloc[0]['ngram']}\" (ratio: {sig_df.iloc[0]['ratio']:.1f}x)\n\n" if len(sig_df) > 0 else ""
        f"## Topics\n"
        f"- {len(topics_df)} attack strategy topics discovered\n\n"
        f"## Sequential Patterns\n"
        f"- {len(patterns_df)} frequent attack sequences found\n\n"
        f"## Detection\n"
        f"- Classifier: {detector.classifier_name}\n"
        f"- Accuracy: {metrics['accuracy']:.3f}\n"
        f"- F1 Score: {metrics['f1']:.3f}\n"
        f"- CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}\n"
    )
    (results_dir / "analysis_summary.md").write_text(summary)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Analysis complete in {elapsed:.1f}s")
    print(f"  Results: {results_dir}/")
    print(f"  Figures: {config['output']['figures_dir']}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_analysis()
