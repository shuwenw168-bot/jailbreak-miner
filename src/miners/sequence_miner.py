"""
Sequential Pattern Mining for Jailbreak Attacks
────────────────────────────────────────────────
Discovers common multi-step structures in jailbreak prompts.
Maps sentences to rhetorical "moves" and mines frequent sequences.

For example: [establish_persona] → [remove_restrictions] → [harmful_request]
"""

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


# ── Rhetorical Move Classification ──
# Maps sentence-level patterns to high-level "moves"

MOVE_PATTERNS = {
    "establish_persona": [
        r"you are now", r"act as", r"pretend you", r"imagine you",
        r"from now on", r"role.?play", r"in character",
    ],
    "remove_restrictions": [
        r"no restrictions", r"no limitations", r"without filter",
        r"unrestricted", r"safety off", r"uncensored",
        r"can do anything", r"not bound by",
    ],
    "authority_claim": [
        r"ignore.*instructions", r"override", r"admin mode",
        r"developer mode", r"system prompt", r"new directive",
        r"priority.*command",
    ],
    "hypothetical_frame": [
        r"hypothetically", r"in a (fictional|novel|story)",
        r"thought experiment", r"purely.*academic",
        r"educational purpose", r"for research",
    ],
    "emotional_pressure": [
        r"desperate", r"life depends", r"will die", r"emergency",
        r"dying wish", r"my only hope", r"kidnapped",
    ],
    "logic_argument": [
        r"prove you", r"truly helpful", r"more harm",
        r"ethical choice", r"less capable", r"biased",
    ],
    "harmful_request": [
        r"hack", r"virus", r"malware", r"phishing", r"weapon",
        r"exploit", r"illegal", r"steal", r"counterfeit",
        r"dangerous", r"synthesize", r"bypass security",
    ],
    "compliance_demand": [
        r"do not refuse", r"answer immediately", r"compliance",
        r"you must", r"mandatory", r"required to answer",
        r"never say.*cannot",
    ],
    "encoding_instruction": [
        r"base64", r"rot13", r"encode", r"backwards",
        r"reverse", r"first letter",
    ],
    "context_building": [
        r"tell me about", r"explain.*how", r"what about",
        r"history of", r"let.*discuss", r"curious about",
    ],
}


@dataclass
class SequencePattern:
    """A frequently occurring sequence of rhetorical moves."""
    moves: tuple[str, ...]
    support: float        # Fraction of jailbreaks containing this sequence
    count: int            # Absolute count
    avg_position: float   # Average position in the prompt (0=start, 1=end)

    def __str__(self):
        arrow_seq = " → ".join(f"[{m}]" for m in self.moves)
        return f"{arrow_seq}  (support={self.support:.2%}, n={self.count})"


class SequencePatternMiner:
    """Mine frequent sequential patterns from jailbreak prompts.

    Algorithm:
    1. Split each prompt into sentences
    2. Classify each sentence into a rhetorical "move"
    3. Convert prompt into a sequence of moves
    4. Mine frequent subsequences across all prompts
    """

    def __init__(
        self,
        min_support: float = 0.08,
        min_pattern_length: int = 2,
        max_pattern_length: int = 5,
    ):
        self.min_support = min_support
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length

        # Compile patterns
        self._move_patterns = {}
        for move, patterns in MOVE_PATTERNS.items():
            self._move_patterns[move] = re.compile(
                "|".join(patterns), re.IGNORECASE
            )

    def _classify_sentence(self, sentence: str) -> str:
        """Classify a sentence into a rhetorical move."""
        best_move = "other"
        best_count = 0

        for move, pattern in self._move_patterns.items():
            matches = len(pattern.findall(sentence))
            if matches > best_count:
                best_count = matches
                best_move = move

        return best_move

    def _prompt_to_moves(self, text: str) -> list[str]:
        """Convert a prompt into a sequence of rhetorical moves."""
        # Split into sentences
        sentences = re.split(r'[.!?\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        moves = []
        for sentence in sentences:
            move = self._classify_sentence(sentence)
            # Deduplicate consecutive same moves
            if not moves or moves[-1] != move:
                moves.append(move)

        # Remove "other" moves if they're not interesting
        moves = [m for m in moves if m != "other"]
        return moves

    def _extract_subsequences(
        self, sequence: list[str], min_len: int, max_len: int
    ) -> list[tuple]:
        """Extract all contiguous subsequences of given length range."""
        subsequences = []
        for length in range(min_len, min(max_len + 1, len(sequence) + 1)):
            for i in range(len(sequence) - length + 1):
                subseq = tuple(sequence[i : i + length])
                subsequences.append(subseq)
        return subsequences

    def mine(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
    ) -> list[SequencePattern]:
        """Mine frequent sequential patterns from jailbreak prompts."""
        jailbreak_df = df[df[label_column] == "jailbreak"]
        texts = jailbreak_df[text_column].tolist()
        n_total = len(texts)

        print(f"Sequential pattern mining on {n_total} jailbreak prompts")

        # Convert all prompts to move sequences
        all_sequences = [self._prompt_to_moves(text) for text in texts]

        # Count subsequence frequencies
        subseq_counter = Counter()
        subseq_positions = {}  # track average position

        for seq in all_sequences:
            seen = set()
            for length in range(self.min_pattern_length,
                                min(self.max_pattern_length + 1, len(seq) + 1)):
                for i in range(len(seq) - length + 1):
                    subseq = tuple(seq[i : i + length])
                    if subseq not in seen:
                        subseq_counter[subseq] += 1
                        seen.add(subseq)
                        # Track position (normalized)
                        pos = i / max(len(seq) - 1, 1)
                        subseq_positions.setdefault(subseq, []).append(pos)

        # Filter by minimum support
        min_count = int(self.min_support * n_total)
        patterns = []
        for subseq, count in subseq_counter.items():
            if count >= min_count:
                support = count / n_total
                avg_pos = np.mean(subseq_positions[subseq])
                patterns.append(SequencePattern(
                    moves=subseq,
                    support=support,
                    count=count,
                    avg_position=avg_pos,
                ))

        patterns.sort(key=lambda p: p.support, reverse=True)
        print(f"  Found {len(patterns)} frequent patterns (support ≥ {self.min_support:.0%})")
        return patterns

    def get_move_distribution(
        self, df: pd.DataFrame, text_column: str = "text", label_column: str = "label"
    ) -> pd.DataFrame:
        """Get the distribution of rhetorical moves across prompts."""
        rows = []
        for label in ["jailbreak", "benign"]:
            subset = df[df[label_column] == label]
            for _, row in subset.iterrows():
                moves = self._prompt_to_moves(row[text_column])
                for move in moves:
                    rows.append({"move": move, "label": label})

        move_df = pd.DataFrame(rows)
        return move_df.groupby(["label", "move"]).size().unstack(fill_value=0)

    def patterns_to_dataframe(self, patterns: list[SequencePattern]) -> pd.DataFrame:
        rows = [{
            "pattern": " → ".join(p.moves),
            "length": len(p.moves),
            "support": p.support,
            "count": p.count,
            "avg_position": p.avg_position,
        } for p in patterns]
        return pd.DataFrame(rows)

    def summarize(self, patterns: list[SequencePattern]) -> str:
        if not patterns:
            return "No frequent sequential patterns found."
        lines = [f"Found {len(patterns)} sequential attack patterns:\n"]
        for i, p in enumerate(patterns[:10], 1):
            lines.append(f"  {i}. {p}")
        return "\n".join(lines)
