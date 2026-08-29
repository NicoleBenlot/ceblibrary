import os

from ceblibrary import Corrector, Decoder

SAMPLE_INDEX = """\
[a]
a = 4
ako = 1
[b]
balay = 7
[c]
can = 1
canoe = 6
[k]
kan = 2
ko = 5
[m]
mo = 2
[n]
nimo = 3
[o]
og = 8
open = 3
[s]
sa = 9
"""


def make_model(tmp_path):
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    (model_dir / "index.txt").write_text(SAMPLE_INDEX, encoding="utf-8")
    (model_dir / "assets").mkdir()
    return str(model_dir)


class TestNormalize:
    def test_vocab_free_lowercases_and_strips_punctuation(self):
        c = Corrector()
        assert c.normalize("AKO, BALAY.") == "ako balay"

    def test_collapses_whitespace(self):
        c = Corrector()
        assert c.normalize("   ako    balay  ") == "ako balay"

    def test_collapses_repeated_letters(self):
        c = Corrector()
        assert c.normalize("hellooo balaaaay") == "hello balay"

    def test_strips_diacritics(self):
        c = Corrector()
        assert c.normalize("baláy ńimo") == "balay nimo"

    def test_drops_filler_words(self):
        c = Corrector()
        assert c.normalize("uh ako um balay") == "ako balay"

    def test_drops_punctuation_only_tokens(self):
        c = Corrector()
        assert c.normalize("ako ... !!! balay") == "ako balay"


class TestCorrection:
    VOCAB = ["a", "ako", "balay", "can", "canoe", "kan", "ko", "mo", "nimo", "og", "open", "sa"]

    def test_known_words_unchanged(self):
        c = Corrector(self.VOCAB)
        assert c.correct("ako balay") == "ako balay"
        assert c.correct("AKO, balay!") == "ako balay"

    def test_auto_correct_near_miss(self):
        c = Corrector(self.VOCAB)
        assert c.correct("akoo balay") == "ako balay"
        assert c.correct("ako balai") == "ako balay"

    def test_unmatched_words_pass_through(self):
        c = Corrector(self.VOCAB)
        assert c.correct("ako skyscraper balay") == "ako skyscraper balay"

    def test_is_known(self):
        c = Corrector(self.VOCAB)
        assert c.is_known("Ako")
        assert not c.is_known("xyz")

    def test_suggest_ranks_nearest_first(self):
        c = Corrector(self.VOCAB)
        assert c.suggest("balay")[0] == "balay"
        assert c.suggest("akoo")[0] == "ako"

    def test_standalone_corrector_without_vocab_only_normalizes(self):
        c = Corrector()
        assert c.correct("AKO, balay!") == "ako balay"
        assert c.correct("akoo balay") == "akoo balay"

    def test_from_decoder(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        c = Corrector.from_decoder(decoder)
        assert c.correct("ako balai") == "ako balay"
        assert c.correct("ako metro balay") == "ako metro balay"

    def test_max_distance_override(self):
        c = Corrector(self.VOCAB, max_distance=0)
        assert c.correct("akoo balay") == "akoo balay"


class TestDecoderWords:
    def test_words_property(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        assert decoder.words == (
            "a", "ako", "balay", "can", "canoe", "kan", "ko",
            "mo", "nimo", "og", "open", "sa",
        )

    def test_words_feeds_corrector(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        c = Corrector(vocabulary=decoder.words)
        assert c.correct("AKO baláy") == "ako balay"