import os
import shutil

import pytest

from ceblibrary import Decoder, ModelNotFoundError

import ceblibrary as _pkg
REPO_MODEL_DIR = os.path.dirname(os.path.dirname(_pkg.__file__))
INDEX_PATH = os.path.join(REPO_MODEL_DIR, "assets", "index.txt")


def _make_copy(tmp_path):
    """Copy the real index.txt into a temp model dir for mutation tests."""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    shutil.copy2(INDEX_PATH, model_dir / "index.txt")
    return str(model_dir)


class TestDecoderLoading:
    def test_loads_by_model_dir(self):
        decoder = Decoder(REPO_MODEL_DIR)
        assert decoder.word_count == 16

    def test_loads_by_index_path(self):
        decoder = Decoder(index_path=INDEX_PATH)
        assert decoder.word_count == 16

    def test_model_dir_property(self):
        decoder = Decoder(REPO_MODEL_DIR)
        assert decoder.model_dir == REPO_MODEL_DIR
        assert decoder.assets_dir == os.path.join(REPO_MODEL_DIR, "assets", "audio")

    def test_available_letters(self):
        decoder = Decoder(REPO_MODEL_DIR)
        letters = decoder.available_letters()
        assert "an" in letters
        assert "ka" in letters
        assert "sa" in letters


class TestDefaultModel:
    def test_default_model_is_repo_root(self):
        decoder = Decoder()
        assert decoder.model_dir == REPO_MODEL_DIR
        assert os.path.isfile(INDEX_PATH)
        assert decoder.word_count > 0

    def test_default_assets_dir(self):
        decoder = Decoder()
        assert decoder.assets_dir == os.path.join(REPO_MODEL_DIR, "assets", "audio")

    def test_missing_model_raises(self):
        with pytest.raises(ModelNotFoundError):
            Decoder("definitely-not-a-model")


class TestWordLookup:
    def test_has_word(self):
        decoder = Decoder()
        assert decoder.has_word("bisan")
        assert decoder.has_word("Bisan")
        assert not decoder.has_word("xyz")

    def test_get_id(self):
        decoder = Decoder()
        assert decoder.get_id("apan") == 11
        assert decoder.get_id("pero") == 12
        assert decoder.get_id("unknown") is None


class TestDecode:
    def test_simple_sentence(self):
        decoder = Decoder()
        assert decoder.decode("pero apan") == [12, 11]

    def test_unknown_words_become_none(self):
        decoder = Decoder()
        assert decoder.decode("pero xyz apan") == [12, None, 11]

    def test_strict_raises_on_unknown(self):
        decoder = Decoder()
        with pytest.raises(KeyError, match="Unknown word"):
            decoder.decode_strict("pero xyz")

    def test_case_insensitive_decode(self):
        decoder = Decoder()
        assert decoder.decode("PERO apan") == [12, 11]


class TestAudioPaths:
    def test_model_default_asset_dir(self):
        decoder = Decoder()
        paths = decoder.audio_paths("pero apan")
        assert len(paths) == 2
        assert paths[0] == os.path.join(decoder.assets_dir, "12.mp3")
        assert paths[1] == os.path.join(decoder.assets_dir, "11.mp3")
        assert os.path.isfile(paths[0])
        assert os.path.isfile(paths[1])

    def test_audio_path_explicit_dir(self):
        decoder = Decoder()
        p = decoder.audio_path(5, os.path.join(REPO_MODEL_DIR, "assets", "audio"))
        assert p.endswith("5.mp3")
        assert os.path.isfile(p)


class TestMutations:
    def test_add_word(self, tmp_path):
        decoder = Decoder(_make_copy(tmp_path))
        decoder.add_word("test", 99)
        assert decoder.get_id("test") == 99

    def test_save_reload(self, tmp_path):
        model_dir = _make_copy(tmp_path)
        decoder = Decoder(model_dir)
        decoder.add_word("newword", 50)
        decoder.save()
        decoder2 = Decoder(model_dir)
        assert decoder2.get_id("newword") == 50

    def test_save_uses_two_letter_sections(self, tmp_path):
        model_dir = _make_copy(tmp_path)
        decoder = Decoder(model_dir)
        decoder.save()
        contents = open(os.path.join(model_dir, "index.txt"), encoding="utf-8").read()
        assert "[an]" in contents
        assert "[bi]" in contents
        assert "[pa]" in contents
        assert "[sa]" in contents