import os
import shutil
import tempfile

import pytest

from ceblibrary import Decoder, ModelNotFoundError

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

# The default model lives at the repo root (parent of the package dir)
# and doubles as the model directory itself.
import ceblibrary as _pkg
REPO_MODEL_DIR = os.path.dirname(os.path.dirname(_pkg.__file__))


def make_model(tmp_path, index_content=SAMPLE_INDEX):
    """Create a standalone model dir (index.txt + assets/) in tmp_path."""
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    (model_dir / "index.txt").write_text(index_content, encoding="utf-8")
    (model_dir / "assets").mkdir()
    return str(model_dir)


class TestDecoderLoading:
    def test_loads_by_model_dir(self, tmp_path):
        model_dir = make_model(tmp_path)
        decoder = Decoder(model_dir)
        assert decoder.word_count == 12

    def test_loads_by_index_path(self, tmp_path):
        model_dir = make_model(tmp_path)
        decoder = Decoder(index_path=os.path.join(model_dir, "index.txt"))
        assert decoder.word_count == 12

    def test_model_dir_property(self, tmp_path):
        model_dir = make_model(tmp_path)
        decoder = Decoder(model_dir)
        assert decoder.model_dir == model_dir
        assert decoder.assets_dir == os.path.join(model_dir, "assets")

    def test_available_letters(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        letters = decoder.available_letters()
        assert "a" in letters
        assert "c" in letters
        assert "s" in letters


class TestDefaultModel:
    def test_default_model_is_repo_root(self):
        if not os.path.isfile(os.path.join(REPO_MODEL_DIR, "index.txt")):
            pytest.skip("no index.txt at repo root")
        decoder = Decoder()
        assert decoder.model_dir == REPO_MODEL_DIR
        assert os.path.isfile(os.path.join(REPO_MODEL_DIR, "index.txt"))
        assert decoder.word_count > 0

    def test_default_assets_dir(self):
        if not os.path.isfile(os.path.join(REPO_MODEL_DIR, "index.txt")):
            pytest.skip("no index.txt at repo root")
        decoder = Decoder()
        assert decoder.assets_dir == os.path.join(REPO_MODEL_DIR, "assets")

    def test_missing_model_raises(self):
        with pytest.raises(ModelNotFoundError):
            Decoder("definitely-not-a-model")


class TestWordLookup:
    def test_has_word(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        assert decoder.has_word("ako")
        assert decoder.has_word("Ako")
        assert not decoder.has_word("xyz")

    def test_get_id(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        assert decoder.get_id("ako") == 1
        assert decoder.get_id("balay") == 7
        assert decoder.get_id("unknown") is None


class TestDecode:
    def test_simple_sentence(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        assert decoder.decode("ako balay") == [1, 7]

    def test_unknown_words_become_none(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        assert decoder.decode("ako xyz balay") == [1, None, 7]

    def test_strict_raises_on_unknown(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        with pytest.raises(KeyError, match="Unknown word"):
            decoder.decode_strict("ako xyz")

    def test_case_insensitive_decode(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        assert decoder.decode("AKO balay") == [1, 7]


class TestAudioPaths:
    def test_model_default_asset_dir(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        paths = decoder.audio_paths("ako balay")
        assert len(paths) == 2
        assert paths[0] == os.path.join(decoder.assets_dir, "1.mp3")
        assert paths[1] == os.path.join(decoder.assets_dir, "7.mp3")

    def test_audio_path_explicit_dir(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        p = decoder.audio_path(5, str(tmp_path))
        assert p.endswith("5.mp3")


class TestMutations:
    def test_add_word(self, tmp_path):
        decoder = Decoder(make_model(tmp_path))
        decoder.add_word("test", 99)
        assert decoder.get_id("test") == 99

    def test_save_reload(self, tmp_path):
        model_dir = make_model(tmp_path)
        decoder = Decoder(model_dir)
        decoder.add_word("newword", 50)
        decoder.save()
        decoder2 = Decoder(model_dir)
        assert decoder2.get_id("newword") == 50
