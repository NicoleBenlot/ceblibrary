import pytest

from ceblibrary import Corrector, Decoder

from tests._assets import WORDS

REAL_WORDS = WORDS


def _near_miss_pair():
    """Pick a (real_word, misspelling) from the model vocab that corrects back.

    Misspellings are built by duplicating a word's first letter, then only
    kept when the Corrector actually maps them home. Data-driven: new vocab
    words are candidates automatically. Skips when the model is too small to
    produce a pair (e.g. an empty or single-word vocabulary).
    """
    for word in REAL_WORDS[:200]:
        miss = word + word[0]
        if Corrector(REAL_WORDS).correct(miss) == word:
            return word, miss
    pytest.skip(
        "model vocabulary {REAL_WORDS!r} has no auto-correctable near-miss "
        "pair; add more words or adjust the correction threshold".format(
            REAL_WORDS=REAL_WORDS
        )
    )


class TestNormalize:
    def test_vocab_free_lowercases_and_strips_punctuation(self):
        c = Corrector()
        assert c.normalize("PERO, BISAN.") == "pero bisan"

    def test_collapses_whitespace(self):
        c = Corrector()
        assert c.normalize("   pero    bisan  ") == "pero bisan"

    def test_collapses_repeated_letters(self):
        c = Corrector()
        assert c.normalize("perooo bisaaaaan") == "pero bisan"

    def test_strips_diacritics(self):
        c = Corrector()
        assert c.normalize("peró bisán") == "pero bisan"

    def test_drops_filler_words(self):
        c = Corrector()
        assert c.normalize("uh pero um bisan") == "pero bisan"

    def test_drops_punctuation_only_tokens(self):
        c = Corrector()
        assert c.normalize("pero ... !!! bisan") == "pero bisan"


class TestCorrection:
    def test_known_words_unchanged(self):
        c = Corrector(REAL_WORDS)
        if not REAL_WORDS:
            pytest.skip("model vocabulary is empty")
        words = " ".join(REAL_WORDS)
        assert c.correct(words) == words
        noisy = ", ".join(
            w.upper() if i % 2 else w for i, w in enumerate(REAL_WORDS)
        )
        assert c.correct(f"{noisy}!") == words

    def test_auto_correct_near_miss(self):
        c = Corrector(REAL_WORDS)
        word, miss = _near_miss_pair()
        assert c.correct(miss) == word
        assert c.correct(miss.upper() + "!") == word

    def test_unmatched_words_pass_through(self):
        c = Corrector(REAL_WORDS)
        word, _ = _near_miss_pair()
        assert c.correct(f"{word} skyscraper {word}") == f"{word} skyscraper {word}"

    def test_is_known(self):
        c = Corrector(REAL_WORDS)
        if not REAL_WORDS:
            pytest.skip("model vocabulary is empty")
        word = REAL_WORDS[0]
        assert c.is_known(word)
        assert c.is_known(word.upper())
        assert not c.is_known("notacebuanoword")

    def test_suggest_exact_word_first(self):
        c = Corrector(REAL_WORDS)
        if not REAL_WORDS:
            pytest.skip("model vocabulary is empty")
        word = REAL_WORDS[0]
        assert c.suggest(word)[0] == word

    def test_suggest_ranks_near_miss_first(self):
        c = Corrector(REAL_WORDS)
        word, miss = _near_miss_pair()
        assert c.suggest(miss)[0] == word

    def test_standalone_corrector_without_vocab_only_normalizes(self):
        c = Corrector()
        assert c.correct("PERO, bisan!") == "pero bisan"
        assert c.correct("peroo bisan") == "peroo bisan"

    def test_from_decoder(self):
        c = Corrector.from_decoder(Decoder())
        word, miss = _near_miss_pair()
        assert c.correct(miss) == word
        assert c.correct(f"{word} skyscraper {word}") == f"{word} skyscraper {word}"

    def test_max_distance_override_blocks_correction(self):
        c = Corrector(REAL_WORDS, max_distance=0)
        word, miss = _near_miss_pair()
        assert c.correct(miss) == miss
        assert c.correct(f"{word} {miss}") == f"{word} {miss}"


class TestDecoderWords:
    def test_words_property_matches_index(self):
        decoder = Decoder()
        assert decoder.words == REAL_WORDS

    def test_words_feeds_corrector(self):
        decoder = Decoder()
        c = Corrector(vocabulary=decoder.words)
        if not REAL_WORDS:
            pytest.skip("model vocabulary is empty")
        words = " ".join(w.upper() if i % 2 else w for i, w in enumerate(REAL_WORDS))
        assert c.correct(words) == " ".join(REAL_WORDS)