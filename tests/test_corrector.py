from ceblibrary import Corrector, Decoder


REAL_WORDS = ("ang", "apan", "aron", "bisan", "hangtud", "kay", "kung", "mao",
              "nga", "og", "para", "pero", "sa", "samtang", "tungod", "ug")


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
        assert c.correct("pero bisan") == "pero bisan"
        assert c.correct("PERO, bisan!") == "pero bisan"

    def test_auto_correct_near_miss(self):
        c = Corrector(REAL_WORDS)
        assert c.correct("peroo bisan") == "pero bisan"
        assert c.correct("pero bisa") == "pero bisan"

    def test_unmatched_words_pass_through(self):
        c = Corrector(REAL_WORDS)
        assert c.correct("pero skyscraper bisan") == "pero skyscraper bisan"

    def test_is_known(self):
        c = Corrector(REAL_WORDS)
        assert c.is_known("Bisan")
        assert not c.is_known("xyz")

    def test_suggest_ranks_nearest_first(self):
        c = Corrector(REAL_WORDS)
        assert c.suggest("bisan")[0] == "bisan"
        assert c.suggest("peroo")[0] == "pero"

    def test_standalone_corrector_without_vocab_only_normalizes(self):
        c = Corrector()
        assert c.correct("PERO, bisan!") == "pero bisan"
        assert c.correct("peroo bisan") == "peroo bisan"

    def test_from_decoder(self):
        decoder = Decoder()
        c = Corrector.from_decoder(decoder)
        assert c.correct("peroo bisan") == "pero bisan"
        assert c.correct("pero skyscraper bisan") == "pero skyscraper bisan"

    def test_max_distance_override(self):
        c = Corrector(REAL_WORDS, max_distance=0)
        assert c.correct("peroo bisan") == "peroo bisan"


class TestDecoderWords:
    def test_words_property(self):
        decoder = Decoder()
        assert decoder.words == REAL_WORDS

    def test_words_feeds_corrector(self):
        decoder = Decoder()
        c = Corrector(vocabulary=decoder.words)
        assert c.correct("PERO bisán") == "pero bisan"