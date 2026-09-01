# AGENTS.md

## Project overview

Cebuano language dictionary/library ("ceblibrary"): a Python library that decodes sentences into audio IDs — a bridge from words to audio. The **repo root doubles as the model directory**: copy the whole repo into an app's `models/` folder, then query it via `Decoder` (Vosk/MMS model pattern).

## Architecture — the repo IS the model

No `models/<name>` subdir. The model files live at the repo root:

```
ceblibrary/            <-- the model dir itself (copy this whole repo to use it)
├── assets/            the dict + audio assets
│   ├── index.txt      word -> audio ID map (sectioned, alphabetical)
│   ├── audio/         audio files named <id>.mp3
│   └── datainfor.txt  design note
├── ceblibrary/        the Python package (code only)
│   ├── __init__.py    exports Decoder, ModelNotFoundError
│   └── decoder.py
├── tests/
├── pyproject.toml
├── README.md          general usage docs
└── AGENTS.md
```

## Model resolution

- `Decoder()` → the repo root (parent of the package) is the default model: `assets/index.txt` and `assets/audio/` are read from there.
- `Decoder("/path/to/model")` → explicit path to any model dir (index found at `<dir>/index.txt` or `<dir>/assets/index.txt`).
- `Decoder(index_path=".../index.txt")` → a specific index file.
- Unknown/missing model raises `ModelNotFoundError`.

So embedding = copying the repo into `<app>/models/ceblibrary/`; `Decoder("CEBLIBRARY_DIR")` or, when it's on the import path, `Decoder()` finds it.

## Shipping model (size)

The wheel ships **code only** — `index.txt` and `assets/` are at the repo root, outside the package, and deliberately not packaged. Apps pull the model dir (from GitHub releases / Vosk-style downloader) separately. Keeps `pip install` tiny; audio assets are fetched on demand.

## Index format

```
[ak]
ako = 1
[ba]
balay = 7
```

Section headers (`[ak]`, `[ba]`, ...), grouped by each word's first two letters, split the word list so lookups can skip unrelated prefixes. Words separated with ` = `. Comments (`#`) and blank lines ignored. Duplicate words: last entry wins.

## Decoder API

- `Decoder(model=None, index_path=None)` — model dir path, or `None` for default (repo root).
- `decode(sentence)` — list of audio IDs per word; unknown words map to `None`.
- `decode_strict(sentence)` — like `decode()`, but raises `KeyError` on unknown words.
- `get_id(word)` / `has_word(word)` — single-word lookup (case-insensitive).
- `audio_paths(sentence, assets_dir=None)` — resolves `assets/audio/<id>.mp3` paths; defaults to the model's own `assets/audio/`.
- `model_dir` / `index_path` / `assets_dir` / `word_count` / `words` — model metadata properties.
- `add_word(word, id)` / `save(path=None)` — build/add entries, write a sectioned index (defaults to the model's `index.txt`).
- Lookups are case-insensitive (words lowercased on load).

## Corrector API (STT normalization/correction)

Used between STT and downstream AI: `user -> stt -> normalize/correct -> cloud AI -> Decoder -> audio`.

- `Corrector(vocabulary=None, *, max_distance=None)` — fully independent of Decoder. No vocabulary = normalization only.
- `Corrector.from_decoder(decoder)` — vocabulary-aware correction using the model's index words (`decoder.words`).
- `normalize(text)` / `normalize_tokens(text)` — vocabulary-free: lowercase, strip diacritics/punctuation, collapse repeated letters, drop fillers.
- `correct(text)` / `correct_tokens(text)` — normalize then map near-miss tokens to the closest known word (edit distance); too-far words pass through untouched.
- `suggest(word, limit=3)` / `is_known(word)` — candidate lookup helpers.

## Development

- **Test:** `python -m pytest -q` (34 tests; requires `pip install pytest`).
- **Run a quick decode:**
  `python -c "from ceblibrary import Decoder; d=Decoder(); print(d.decode('ako mo open sa balay'))"`
- No packaging/lint/typecheck/CI configured. `pyproject.toml` is minimal pyproject-only (setuptools legacy backend) for pip installability; package contains only `ceblibrary/` code (no bundled model).

## Gotchas

- **Keep the package dir named `ceblibrary/`** (flat layout, alongside `index.txt`/`assets/`). The default model resolution relies on the package being a direct child of the model root (`dirname(dirname(decoder.py))`). Renaming it to `src/` breaks `import ceblibrary` and the test suite; a previous attempt at a src-layout could not be imported in this Python 3.14 environment.
