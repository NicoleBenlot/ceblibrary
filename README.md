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
d.audio_paths("ako mo open sa balay")    # -> [.../assets/audio/1.mp3, .../assets/audio/7.mp3]
```

Unknown words decode to `None`; `decode_strict` raises `KeyError` instead.

## The repo is the model

```
ceblibrary/            model directory — copy this whole repo to embed it
├── index.txt          word -> audio ID map (sectioned, alphabetical)
├── assets/audio/      audio clips named <id>.mp3
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
[ak]
ako = 1
[ba]
balay = 7
```

The `[ak]`, `[ba]`, … section headers (grouped by each word's first two letters) let lookups skip unrelated prefixes instead of scanning the whole list. Words use ` = ` separators; `#` comments and blank lines are ignored; duplicate words resolve to the last entry.

## API

- `Decoder(model=None, index_path=None)`
- `decode(sentence)` → list of audio IDs per word (`None` for unknown)
- `decode_strict(sentence)` → same, but raises `KeyError` on unknown words
- `get_id(word)` / `has_word(word)` — single-word lookup (case-insensitive)
- `audio_paths(sentence, assets_dir=None)` → `assets/audio/<id>.mp3` paths
- `add_word(word, id)` / `save(path=None)` — build and write a sectioned index
- `model_dir` / `index_path` / `assets_dir` / `word_count` / `words` — model metadata
- `Corrector(vocabulary=None, max_distance=None)` / `Corrector.from_decoder(decoder)` — clean up STT output
- `Corrector.normalize(text)` — vocabulary-free text cleaning (lowercase, punctuation, diacritics, repeated letters, fillers)
- `Corrector.correct(text)` — normalize + map mistyped/misheard words to the closest known word
- `Corrector.normalize_tokens` / `correct_tokens` / `suggest(word)` / `is_known(word)`

**STT cleanup pipeline** — STT mishears Cebuano words; normalize/correct before forwarding to downstream AI:

```python
from ceblibrary import Corrector, Decoder

decoder = Decoder()
corrector = Corrector.from_decoder(decoder)   # or Corrector() for normalize-only
cleaned = corrector.correct("uh AKO, balaay.")  # -> "ako balay"
```

Words too far from the dictionary pass through untouched, leaving the cloud AI to interpret them. `Corrector` is fully independent of `Decoder` — it needs no model unless you want vocabulary-aware correction.

## Development

```sh
pip install pytest
python -m pytest -q   # 34 tests
```
