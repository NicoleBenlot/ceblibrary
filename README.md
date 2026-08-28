# ceblibrary

A Cebuano language dictionary-to-audio library. It decodes a sentence into a sequence of word audio IDs — a bridge from words to spoken Cebuano.

The **repo itself doubles as the model directory**: copy the whole repo into an app's `models/` folder and query it with `Decoder`, following the Vosk / MMS model pattern.

## What it does

Given a sentence, `Decoder` maps each word to its audio ID:

```python
from ceblibrary import Decoder

d = Decoder()                       # repo root is the default model
d.decode("Can you open a can?")     # -> [1, None, 3, 4, None]
d.decode_strict("ako mo open sa balay")  # -> [1, 2, 3, 9, 7]
d.audio_paths("ako mo open sa balay")    # -> [.../assets/1.mp3, .../assets/7.mp3]
```

Unknown words decode to `None`; `decode_strict` raises `KeyError` instead.

## The repo is the model

```
ceblibrary/            model directory — copy this whole repo to embed it
├── index.txt          word -> audio ID map (sectioned, alphabetical)
├── assets/            audio clips named <id>.mp3
├── ceblibrary/        the Python package (code only)
├── tests/
└── pyproject.toml
```

- `Decoder()` — uses the repo root (parent of the package) as the model.
- `Decoder("/path/to/model")` — explicit model directory.
- `Decoder(index_path="…/index.txt")` — a specific index file.

Place the copied repo at `<app>/models/ceblibrary/`; `Decoder("…/ceblibrary")` or, when it's importable, `Decoder()` finds it automatically.

## Installing

```sh
pip install .
```

The wheel ships **code only** — the small index and audio assets sit at the repo root and are intentionally not bundled, keeping the install tiny. Copy the model directory into the app and point `Decoder` at it (or pull it from a release on demand).

## Index format

```
[a]
ako = 1
[b]
balay = 7
```

The `[a]`, `[b]`, … section headers let lookups skip unrelated prefixes instead of scanning the whole list. Words use ` = ` separators; `#` comments and blank lines are ignored; duplicate words resolve to the last entry.

## API

- `Decoder(model=None, index_path=None)`
- `decode(sentence)` → list of audio IDs per word (`None` for unknown)
- `decode_strict(sentence)` → same, but raises `KeyError` on unknown words
- `get_id(word)` / `has_word(word)` — single-word lookup (case-insensitive)
- `audio_paths(sentence, assets_dir=None)` → `assets/<id>.mp3` paths
- `add_word(word, id)` / `save(path=None)` — build and write a sectioned index
- `model_dir` / `index_path` / `assets_dir` / `word_count` — model metadata

## Development

```sh
pip install pytest
python -m pytest -q   # 17 tests
```
