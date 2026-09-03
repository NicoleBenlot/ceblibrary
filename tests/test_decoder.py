import os
import shutil

import pytest

from ceblibrary import Decoder, ModelNotFoundError

from tests._assets import (
    AUDIO_DIR,
    INDEX,
    INDEX_PATH,
    REPO_MODEL_DIR,
    SECTIONS,
    WORD_COUNT,
    WORDS,
    word_section,
)


def _make_copy(tmp_path):
    """Copy the real index.txt into a temp model dir for mutation tests."""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    shutil.copy2(INDEX_PATH, model_dir / "index.txt")
    return str(model_dir)


class TestDecoderLoading:
    def test_loads_by_model_dir(self):
        assert Decoder(REPO_MODEL_DIR).word_count == WORD_COUNT

    def test_loads_by_index_path(self):
        assert Decoder(index_path=INDEX_PATH).word_count == WORD_COUNT

    def test_model_dir_property(self):
        decoder = Decoder(REPO_MODEL_DIR)
        assert decoder.model_dir == REPO_MODEL_DIR
        assert decoder.assets_dir == AUDIO_DIR

    def test_available_letters_matches_index_sections(self):
        decoder = Decoder(REPO_MODEL_DIR)
        assert set(decoder.available_letters()) == SECTIONS


class TestDefaultModel:
    def test_default_model_is_repo_root(self):
        decoder = Decoder()
        assert decoder.model_dir == REPO_MODEL_DIR
        assert os.path.isfile(INDEX_PATH)
        assert decoder.word_count == WORD_COUNT

    def test_default_assets_dir(self):
        assert Decoder().assets_dir == AUDIO_DIR

    def test_missing_model_raises(self):
        with pytest.raises(ModelNotFoundError):
            Decoder("definitely-not-a-model")


class TestWordLookup:
    def test_has_known_word_case_insensitive(self):
        if not WORDS:
            pytest.skip("model vocabulary is empty")
        decoder = Decoder()
        for word in WORDS[:25]:
            assert decoder.has_word(word)
            assert decoder.has_word(word.upper())

    def test_has_unknown_word(self):
        assert not Decoder().has_word("notacebuanoword")

    def test_get_id_matches_index(self):
        decoder = Decoder()
        for word, audio_id in INDEX.items():
            assert decoder.get_id(word) == audio_id

    def test_get_id_is_case_insensitive(self):
        decoder = Decoder()
        for word in WORDS:
            assert decoder.get_id(word) == decoder.get_id(word.upper())

    def test_get_id_unknown_is_none(self):
        assert Decoder().get_id("notacebuanoword") is None


class TestDecode:
    def test_simple_sentence(self):
        decoder = Decoder()
        sample = WORDS[:100]
        assert decoder.decode(" ".join(sample)) == [INDEX[w] for w in sample]

    def test_unknown_words_become_none(self):
        if not WORDS:
            pytest.skip("model vocabulary is empty")
        decoder = Decoder()
        word = WORDS[0]
        assert decoder.decode(f"{word} xyz {word}") == [INDEX[word], None, INDEX[word]]

    def test_strict_raises_on_unknown(self):
        if not WORDS:
            pytest.skip("model vocabulary is empty")
        with pytest.raises(KeyError, match="Unknown word"):
            Decoder().decode_strict(f"{WORDS[0]} xyz")

    def test_case_insensitive_decode(self):
        if not WORDS:
            pytest.skip("model vocabulary is empty")
        decoder = Decoder()
        sample = WORDS[:50]
        noisy = " ".join(w.upper() if i % 2 else w for i, w in enumerate(sample))
        assert decoder.decode(noisy) == [INDEX[w] for w in sample]


class TestAudioPaths:
    def test_model_default_asset_dir(self):
        if not WORDS:
            pytest.skip("model vocabulary is empty")
        decoder = Decoder()
        sample = WORDS[:50]
        paths = decoder.audio_paths(" ".join(sample))
        assert len(paths) == len(sample)
        for word, path in zip(sample, paths):
            assert path == os.path.join(
                AUDIO_DIR, f"{INDEX[word]}.{decoder.audio_format}"
            )
            assert os.path.isfile(path)

    def test_every_index_id_has_an_audio_file(self):
        decoder = Decoder()
        for word, audio_id in INDEX.items():
            path = decoder.audio_path(audio_id, AUDIO_DIR)
            assert os.path.isfile(path), f"missing audio for {word!r}: {path}"

    def test_audio_path_explicit_dir(self):
        if not WORDS:
            pytest.skip("model vocabulary is empty")
        decoder = Decoder()
        audio_id = INDEX[WORDS[0]]
        path = decoder.audio_path(audio_id, AUDIO_DIR)
        assert path == os.path.join(AUDIO_DIR, f"{audio_id}.{decoder.audio_format}")
        assert os.path.isfile(path)


class TestMutations:
    def test_add_word(self, tmp_path):
        decoder = Decoder(_make_copy(tmp_path))
        decoder.add_word("test", 99)
        assert decoder.get_id("test") == 99

    def test_save_reload_preserves_index(self, tmp_path):
        model_dir = _make_copy(tmp_path)
        decoder = Decoder(model_dir)
        decoder.add_word("newword", 50)
        decoder.save()
        decoder2 = Decoder(model_dir)
        assert decoder2.get_id("newword") == 50
        for word, audio_id in INDEX.items():
            assert decoder2.get_id(word) == audio_id

    def test_save_writes_index_sections(self, tmp_path):
        model_dir = _make_copy(tmp_path)
        decoder = Decoder(model_dir)
        decoder.save()
        contents = open(os.path.join(model_dir, "index.txt"), encoding="utf-8").read()
        for word in WORDS:
            assert f"[{word_section(word)}]" in contents