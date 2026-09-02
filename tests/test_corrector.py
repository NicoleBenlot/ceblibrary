import pytest

from ceblibrary import Corrector, Decoder

from tests._assets import WORDS

REAL_WORDS = WORDS


def _near_miss_pair():
    """Pick a (real_word, misspelling) from the model vocab that corrects back.

    Misspellings are built by duplicating a word's first letter, then only
    kept when the Corrector actually maps them home. Data-driven: new vocab
    words are candidates automatically.
    """
    for word in REAL_WORDS:
        miss = word + word[0]
        if Corrector(REAL_WORDS).correct(miss) == word:
            return word, miss
    pytest.fail(
        f"no word in the model vocabulary {REAL_WORDS!r} corrects back from a "
        "suffixed near-miss; update _near_miss_pair() or the vocabulary"
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
        w1, w2 = REAL_WORDS[0], REAL_WORDS[1]
        assert c.correct(f"{w1} {w2}") == f"{w1} {w2}"
        assert c.correct(f"{w1.upper()}, {w2}!") == f"{w1} {w2}"

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
        assert c.is_known(REAL_WORDS[0])
        assert c.is_known(REAL_WORDS[0].upper())
        assert not c.is_known("notacebuanoword")

    def test_suggest_exact_word_first(self):
        c = Corrector(REAL_WORDS)
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
        w1, w2 = REAL_WORDS[0], REAL_WORDS[1]
        assert c.correct(f"{w1.upper()} {w2}") == f"{w1} {w2}"