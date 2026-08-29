import re
import unicodedata
from typing import Iterable, List, Optional

_REPEATED = re.compile(r"([a-z])\1{2,}")


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two words (bounded by length difference)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


class Corrector:
    """Normalizes and corrects noisy STT output before it reaches a cloud AI.

    Two independent layers:

    - normalize()  — vocabulary-free text cleaning: lowercase, diacritic
      stripping, punctuation/whitespace cleanup, repeated-letter collapse
      ("hellooo" -> "hello"), and filler-word removal (uh, um, ...).
    - correct()    — normalize(), then vocabulary-aware near-miss correction:
      mistyped/misheard tokens are mapped to the closest known word by edit
      distance; tokens too far from any known word pass through untouched,
      leaving the downstream AI to decide.

    Corrector works standalone (normalization only) or with a word list::

        c = Corrector()                    # no vocab -> normalization only
        c = Corrector(decoder.words)       # vocab straight from the dictionary
        c = Corrector.from_decoder(decoder)
    """

    fillers = frozenset(
        {"uh", "um", "ah", "uhm", "er", "erm", "eh", "hmm", "huh", "like"}
    )

    def __init__(
        self,
        vocabulary: Optional[Iterable[str]] = None,
        *,
        max_distance: Optional[int] = None,
    ):
        """vocabulary is any iterable of known words (lowercased on load).

        max_distance overrides the auto correction threshold (1 for words up
        to 4 letters, 2 for longer words).
        """
        self._max_distance = max_distance
        self._words: List[str] = []
        self._known: set = set()
        if vocabulary is not None:
            self._words = sorted({w.strip().lower() for w in vocabulary if w.strip()})
            self._known = set(self._words)

    @classmethod
    def from_decoder(cls, decoder) -> "Corrector":
        """Build a Corrector whose vocabulary is a Decoder's index words."""
        return cls(vocabulary=decoder.words)

    # -- normalization (vocabulary-free) -------------------------------------

    def normalize(self, text: str) -> str:
        return " ".join(self.normalize_tokens(text))

    def normalize_tokens(self, text: str) -> List[str]:
        tokens = []
        for token in text.lower().split():
            token = unicodedata.normalize("NFKD", token)
            token = "".join(ch for ch in token if not unicodedata.combining(ch))
            token = "".join(ch for ch in token if ch.isalpha())
            if not token:
                continue
            token = _REPEATED.sub(r"\1", token)
            if token in self.fillers:
                continue
            tokens.append(token)
        return tokens

    # -- correction (vocabulary-aware) -----------------------------------------

    def correct(self, text: str) -> str:
        return " ".join(self.correct_tokens(text))

    def correct_tokens(self, text: str) -> List[str]:
        return [self._correct_word(w) for w in self.normalize_tokens(text)]

    def _correct_word(self, token: str) -> str:
        if not self._known or token in self._known:
            return token
        best = self._nearest(token)
        return best if best is not None else token

    def _nearest(self, token: str) -> Optional[str]:
        threshold = (
            self._max_distance
            if self._max_distance is not None
            else (1 if len(token) <= 4 else 2)
        )
        best: Optional[str] = None
        best_dist = threshold + 1
        for word in self._words:
            dist = _levenshtein(token, word)
            if dist < best_dist:
                best_dist = dist
                best = word
        return best if best_dist <= threshold else None

    def suggest(self, word: str, limit: int = 3) -> List[str]:
        """Closest vocabulary words to a given word, best-first."""
        target = word.strip().lower()
        scored = sorted(
            self._words,
            key=lambda w: (_levenshtein(target, w), abs(len(w) - len(target)), w),
        )
        return scored[:limit]

    def is_known(self, word: str) -> bool:
        return word.strip().lower() in self._known