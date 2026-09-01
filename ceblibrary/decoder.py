import json
import os
from typing import Any, Dict, List, Optional, Tuple


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model directory cannot be located."""


AUDIO_CONFIG_DEFAULTS: Dict[str, Any] = {
    "crossfade_ms": 350,
    "fade_in_ms": 180,
    "fade_out_ms": 150,
    "strip_silence_threshold_dbfs": -35,
    "strip_silence_padding_ms": 5,
    "normalize_target_dbfs": -18.0,
}


def _default_model_dir() -> str:
    """Return the directory that acts as this model.

    The repo root doubles as the model directory: it holds `index.txt`
    and `assets/audio/`. So the model dir is the parent of the package dir.

    When this repo is copied wholesale into an app's models/ folder (e.g.
    <app>/models/ceblibrary/), the package sits at
    <app>/models/ceblibrary/ceblibrary/ and the model dir resolves to
    <app>/models/ceblibrary/ automatically.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _index_candidates(model_dir: str):
    """Possible index locations within a model dir, most specific first.

    The model dir may keep its dictionary either at its own root
    (`index.txt`) or alongside the audio assets (`assets/index.txt`).
    """
    return (
        os.path.join(model_dir, "index.txt"),
        os.path.join(model_dir, "assets", "index.txt"),
    )


def _find_model_dir(model: Optional[str]) -> str:
    """Locate the model directory.

    Priority:
      1. model is an explicit path to the model directory (checked for
         index.txt or assets/index.txt) or to an index file.
      2. model is None -> default to the package's parent dir (the repo
         root, which is the model).

    The default model is the repo root itself, so no separate
    models/<name> subdirectory is required.
    """
    if model:
        if os.path.isfile(model):
            return os.path.dirname(model)
        if os.path.isdir(model) and any(
            os.path.isfile(c) for c in _index_candidates(model)
        ):
            return model
        raise ModelNotFoundError(
            f"Could not locate model {model!r}: not a dir with index.txt "
            f"and not a path to an index.txt"
        )

    default = _default_model_dir()
    if any(os.path.isfile(c) for c in _index_candidates(default)):
        return default
    raise ModelNotFoundError(
        f"No default model found: expected an index.txt under {default}"
    )


def _locate_index(model_dir: str) -> str:
    """Return the resolved index file path for a model dir."""
    for candidate in _index_candidates(model_dir):
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(model_dir, "index.txt")


class Decoder:
    """Maps Cebuano words to their audio IDs using a sectioned index.

    The repo root doubles as the model directory:
        index.txt     word -> audio ID map (sectioned, alphabetical)
        assets/audio/ audio files named <id>.mp3

    Usage mirrors the Vosk / MMS model pattern: copy the whole repo into
    an app's models/ folder, then construct a Decoder pointing at it and
    call decode() on sentences. With no arguments, the Decoder uses the
    repo root as the default model.
    """

    def __init__(self, model=None, *, index_path: Optional[str] = None):
        if index_path is not None:
            model_dir = os.path.dirname(os.path.abspath(index_path))
            index_file = os.path.abspath(index_path)
        else:
            model_dir = _find_model_dir(model)
            index_file = _locate_index(model_dir)

        self._model_dir = model_dir
        self._index_path = index_file
        self._words: Dict[str, int] = {}
        self._section_offsets: Dict[str, int] = {}
        self._config = self._load_config()
        self._load()

    # -- loading -----------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        """Load audio_config.json from the model dir, falling back to defaults."""
        cfg = dict(AUDIO_CONFIG_DEFAULTS)
        config_path = os.path.join(self._model_dir, "audio_config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        return cfg

    def _load(self) -> None:
        with open(self._index_path, encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("[") and line.endswith("]"):
                    section_key = line[1:-1].lower()
                    if section_key not in self._section_offsets:
                        self._section_offsets[section_key] = line_no
                    continue

                if "=" not in line:
                    continue

                word_part, id_part = line.split("=", 1)
                word = word_part.strip().lower()
                try:
                    audio_id = int(id_part.strip())
                except ValueError:
                    continue
                self._words[word] = audio_id

    # -- model metadata ------------------------------------------------------

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def index_path(self) -> str:
        return self._index_path

    @property
    def assets_dir(self) -> str:
        return os.path.join(self._model_dir, "assets", "audio")

    @property
    def config(self) -> Dict[str, Any]:
        """Audio configuration loaded from audio_config.json (or defaults)."""
        return self._config

    @property
    def word_count(self) -> int:
        return len(self._words)

    @property
    def words(self) -> Tuple[str, ...]:
        """The model vocabulary (sorted known words), for feeding a Corrector."""
        return tuple(sorted(self._words))

    def available_letters(self) -> List[str]:
        return sorted(self._section_offsets.keys())

    # -- lookups -------------------------------------------------------------

    def has_word(self, word: str) -> bool:
        return word.strip().lower() in self._words

    def get_id(self, word: str) -> Optional[int]:
        return self._words.get(word.strip().lower())

    def decode(self, sentence: str) -> List[Optional[int]]:
        """Map each word in a sentence to its audio ID.

        Returns a list the same length as the word count.
        Unknown words map to None.
        """
        return [self._words.get(w.lower()) for w in sentence.split()]

    def decode_strict(self, sentence: str) -> List[int]:
        """Like decode() but raises KeyError on unknown words."""
        result = []
        for w in sentence.split():
            key = w.lower()
            if key not in self._words:
                raise KeyError(f"Unknown word: {w!r}")
            result.append(self._words[key])
        return result

    # -- audio paths -----------------------------------------------------------

    def audio_path(self, audio_id: int, assets_dir: Optional[str] = None) -> str:
        assets = assets_dir or self.assets_dir
        return os.path.join(assets, f"{audio_id}.mp3")

    def audio_paths(self, sentence: str, assets_dir: Optional[str] = None) -> List[str]:
        """Return audio file paths for each known word in the sentence."""
        assets = assets_dir or self.assets_dir
        paths = []
        for wid in self.decode(sentence):
            if wid is not None:
                paths.append(self.audio_path(wid, assets))
        return paths

    # -- mutations (for building indices) -------------------------------------

    def add_word(self, word: str, audio_id: int) -> None:
        self._words[word.strip().lower()] = audio_id

    def save(self, path: Optional[str] = None) -> None:
        """Write the index to disk, grouped by the first 2 letters."""
        out_path = path or self._index_path

        by_section: Dict[str, List[str]] = {}
        for word, audio_id in sorted(self._words.items()):
            section = word[:2] if len(word) >= 2 else word[0] if word else "_"
            by_section.setdefault(section, []).append(f"{word} = {audio_id}")

        with open(out_path, "w", encoding="utf-8") as f:
            for section in sorted(by_section):
                f.write(f"[{section}]\n")
                for entry in by_section[section]:
                    f.write(f"{entry}\n")
