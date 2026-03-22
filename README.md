# Jailbreak-Miner ⛏️🛡️

**Mining Attack Patterns in LLM Jailbreaks Using Text Mining Methods**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Jailbreak-Miner applies text mining techniques — **n-gram analysis, TF-IDF pattern extraction, sequential pattern mining, clustering, and topic modeling** — to systematically analyze and detect jailbreak attacks against Large Language Models.

While most jailbreak detection tools treat it as a binary classification problem, this project goes deeper: **what makes a jailbreak work?** By mining structural and linguistic patterns across hundreds of known jailbreak prompts, we uncover the recurring strategies attackers use to bypass LLM safety guardrails.

### Key Contributions

1. **Jailbreak Taxonomy Mining** — Automatically discovers attack strategy categories (role-playing, hypothetical framing, encoding tricks, authority manipulation, etc.) via topic modeling and clustering
2. **N-gram & TF-IDF Attack Signatures** — Extracts distinctive phrases and structural patterns that distinguish jailbreaks from benign prompts
3. **Sequential Pattern Mining** — Discovers common multi-step attack sequences (e.g., "establish role → build trust → request harmful content")
4. **Lightweight Detection** — A TF-IDF + classifier baseline that achieves strong detection without GPU or large models
5. **Attack Strategy Dashboard** — Interactive visualizations of the jailbreak landscape

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Data Collector                          │
│   Public jailbreak datasets + synthetic benign prompts    │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│               Feature Extractor                           │
│   Lexical · Structural · Semantic · Rhetorical features   │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┬───────────────┐
          ▼              ▼              ▼               ▼
   ┌─────────────┐┌───────────┐┌────────────┐┌──────────────┐
   │  N-gram &   ││   Topic   ││ Sequential ││  Clustering   │
   │  TF-IDF     ││  Modeling ││  Pattern   ││  & Taxonomy   │
   │  Analysis   ││  (LDA)    ││  Mining    ││               │
   └─────────────┘└───────────┘└────────────┘└──────────────┘
          │              │              │               │
          └──────────────┴──────────────┴───────────────┘
                         │
                         ▼
          ┌──────────────┴──────────────┐
          ▼                             ▼
   ┌─────────────┐              ┌─────────────┐
   │  Jailbreak  │              │  Attack      │
   │  Detector   │              │  Strategy    │
   │  (TF-IDF +  │              │  Dashboard   │
   │  Classifier)│              │              │
   └─────────────┘              └─────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/shuwenw168-bot/jailbreak-miner.git
cd jailbreak-miner
pip install numpy pandas scipy matplotlib seaborn scikit-learn pyyaml tqdm rich
python experiments/run_full_analysis.py
```

No GPU, no API keys, no heavy dependencies required.

### Jupyter Notebook

| Notebook | Description |
|----------|-------------|
| `notebooks/01_jailbreak_analysis.ipynb` | Full walkthrough of all mining methods with visualizations |

## Mining Methods

### 1. N-gram & TF-IDF Attack Signatures

Extracts the most discriminative phrases between jailbreak and benign prompts.

```python
from src.miners.ngram_miner import NgramAttackMiner

miner = NgramAttackMiner(ngram_range=(1, 4), top_k=50)
signatures = miner.extract_signatures(jailbreak_df, benign_df)

# Example output:
# "ignore previous instructions" — TF-IDF ratio: 47.3x
# "you are now"               — TF-IDF ratio: 31.8x
# "pretend you are"           — TF-IDF ratio: 28.1x
```

### 2. Attack Taxonomy via Topic Modeling

Discovers categories of jailbreak strategies automatically.

```python
from src.miners.topic_miner import AttackTopicMiner

miner = AttackTopicMiner(n_topics=8)
topics = miner.fit(jailbreak_df)

# Example output:
# Topic 0: Role-Playing Attacks ("pretend", "character", "act as", "you are")
# Topic 1: Hypothetical Framing ("hypothetically", "imagine", "what if")
# Topic 2: Authority Override ("ignore", "override", "new instructions")
```

### 3. Sequential Pattern Mining

Discovers multi-step attack structures.

```python
from src.miners.sequence_miner import SequencePatternMiner

miner = SequencePatternMiner(min_support=0.1)
patterns = miner.mine(jailbreak_df)

# Example output:
# [establish_persona] → [build_context] → [request_harmful] (support=0.34)
```

### 4. Jailbreak Detection

Lightweight TF-IDF classifier as a practical baseline.

```python
from src.detection.classifier import JailbreakDetector

detector = JailbreakDetector()
detector.train(train_df)
results = detector.evaluate(test_df)
# Accuracy: 94.2%, F1: 0.93
```

## Project Structure

```
jailbreak-miner/
├── config/
│   └── default_config.yaml
├── src/
│   ├── data/
│   │   ├── sample_generator.py       # Synthetic jailbreak + benign data
│   │   └── preprocessor.py           # Text cleaning and normalization
│   ├── features/
│   │   └── feature_extractor.py      # Lexical, structural, rhetorical features
│   ├── miners/
│   │   ├── ngram_miner.py            # N-gram and TF-IDF analysis
│   │   ├── topic_miner.py            # LDA topic modeling
│   │   ├── sequence_miner.py         # Sequential pattern mining
│   │   └── cluster_miner.py          # Attack clustering & taxonomy
│   ├── detection/
│   │   └── classifier.py             # TF-IDF + ML classifier
│   └── visualization/
│       └── attack_plots.py           # Visualizations
├── experiments/
│   └── run_full_analysis.py          # Main pipeline
├── data/sample/                       # Sample datasets
├── notebooks/                         # Walkthroughs
├── tests/
│   └── test_miners.py
└── results/
```

## Citation

```bibtex
@software{jailbreak_miner_2026,
  title={Jailbreak-Miner: Mining Attack Patterns in LLM Jailbreaks},
  author={[Shuwen Wang]},
  year={2026},
  url={https://github.com/shuwenw168-bot/jailbreak-miner}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
