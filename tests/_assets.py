"""Test-side reference data parsed from the real model assets.

Tests should assert against THIS data (the actual `assets/index.txt` +
`assets/audio/`), never against hard-coded words/IDs. When the model
grows or the audio set changes, the tests keep working and enforce that
the Decoder and the shipped assets stay in sync.
"""

import os

import ceblibrary as _pkg

REPO_MODEL_DIR = os.path.dirname(os.path.dirname(_pkg.__file__))
INDEX_PATH = os.path.join(REPO_MODEL_DIR, "assets", "index.txt")
AUDIO_DIR = os.path.join(REPO_MODEL_DIR, "assets", "audio")


def _parse_index(path):
    """Parse an index.txt the same way the Decoder does (last entry wins)."""
    words = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            if "=" not in line:
                continue
            word_part, id_part = line.split("=", 1)
            try:
                words[word_part.strip().lower()] = int(id_part.strip())
            except ValueError:
                continue
    return words


def _parse_sections(path):
    sections = set()
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                sections.add(line[1:-1].lower())
    return sections


INDEX = _parse_index(INDEX_PATH)
SECTIONS = _parse_sections(INDEX_PATH)
WORDS = tuple(sorted(INDEX))
WORD_COUNT = len(INDEX)


def word_section(word):
    """The [section] a word belongs to (mirrors Decoder.save's grouping)."""
    return word[:2] if len(word) >= 2 else word[0] if word else "_"