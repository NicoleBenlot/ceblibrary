# ceblibrary

A Cebuano language dictionary-to-audio library. It decodes a sentence into a sequence of word audio IDs — a bridge from words to spoken Cebuano.

The **repo itself doubles as the model directory**: copy the whole repo into an app's `models/` folder and query it with `Decoder`, following the Vosk / MMS model pattern.

## What it does

Given a sentence, `Decoder` maps each word to its audio ID:

```python
from ceblibrary import Decoder

d = Decoder()                       # repo root is the default model
d.decode("pero bisan apan")         # -> [12, 14, 11]
d.decode_strict("pero bisan apan")  # -> [12, 14, 11]
d.audio_paths("pero bisan apan")    # -> [.../assets/audio/12.opus, .../assets/audio/14.opus, .../assets/audio/11.opus]
```

Unknown words decode to `None`; `decode_strict` raises `KeyError` instead.

## The repo is the model

```
ceblibrary/            model directory — copy this whole repo to embed it
├── assets/
│   ├── index.txt      word -> audio ID map (sectioned, alphabetical)
│   └── audio/         audio clips named <id>.<audio_format>
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

## Demo — hear it before wiring it in

`demo.py` decodes a sentence against the real model (`assets/index.txt` + `assets/audio/`), concatenates the word clips, and plays them. Requires `pydub` and `sounddevice` (plus an audio backend for pydub, e.g. `ffmpeg`).

```sh
python demo.py "pero bisan apan"              # decode + play
python demo.py "pero bisan apan" -o out.mp3   # also save an mp3
python demo.py --stream "pero bisan apan"     # streaming/pipelined playback
```

## Streaming / pipelined playback

`StreamPlayer` plays a sentence with a producer/consumer pipeline instead of
evaluating, fetching, and joining everything before starting audio:

```text
sentence
   |  producer thread: decode word -> id (incremental)
   v
prefetch (bounded, LRU-cached)  ->  ordered buffer  ->  playback feeds the device
```

First audio starts as soon as the first word's clip is decoded and loaded;
later words are evaluated and prefetched while playback runs. Audio is always
pushed in sentence order (sequence-numbered), so a faster-loaded later clip
never overtakes an earlier one.

```python
from ceblibrary import Decoder, StreamPlayer

player = StreamPlayer(Decoder(), prefetch_limit=4, cache_size=64)
player.play("pero bisan apan")   # blocks until done; unknown/failed clips skipped
```

- `prefetch_limit` (default 4): max clips loading concurrently — the prefetch
  buffer depth.
- `cache_size` (default 64): LRU cache of recently decoded clips, so repeated
  words don't reload from disk.
- `play(sentence)` (blocking) / `play_async(sentence)` (returns a
  `threading.Event`) / `stop()` (cancel).
- Missing or unreadable clips are logged and skipped — they never crash the
  whole sentence.
- `sink=` overrides the output device (used for testing).

## Shrinking the model: MP3 → Opus

The embedded audio is the biggest part of the model. For speech, Opus at
48 kbps is roughly **half the size of MP3** at the same perceived quality, with
no re-recording — just transcode the existing clips (needs ffmpeg):

```sh
python convert_audio.py --delete-mp3     # *.mp3 -> *.opus, then remove mp3s
```

Then point the decoder at the new extension in `audio_config.json` (repo root):

```json
{ "audio_format": "opus" }
```

The decoder resolves `assets/audio/<id>.<audio_format>`, so the dictionary and
word→ID mappings are unchanged. (`Decoder.audio_format` defaults to `"mp3"`.)

## Index format

```
[bi]
bisan = 14
[pa]
pan = 10
pero = 12
```

The `[bi]`, `[pa]`, … section headers (grouped by each word's first two letters) let lookups skip unrelated prefixes instead of scanning the whole list. Words use ` = ` separators; `#` comments and blank lines are ignored; duplicate words resolve to the last entry.

## API

Full user's guide with runnable examples, a complete call reference, and a
pipeline walkthrough: **[docs/API.md](docs/API.md)**.

- `Decoder(model=None, index_path=None)`
- `decode(sentence)` → list of audio IDs per word (`None` for unknown)
- `decode_strict(sentence)` → same, but raises `KeyError` on unknown words
- `get_id(word)` / `has_word(word)` — single-word lookup (case-insensitive)
- `audio_paths(sentence, assets_dir=None)` → `assets/audio/<id>.<fmt>` paths
- `audio_format` — the clip extension used (`"mp3"` default; set to `"opus"` in config)
- `add_word(word, id)` / `save(path=None)` — build and write a sectioned index
- `model_dir` / `index_path` / `assets_dir` / `word_count` / `words` — model metadata
- `StreamPlayer(decoder, prefetch_limit=4, cache_size=64, sink=None)` — pipelined playback
- `StreamPlayer.play(sentence)` / `play_async(sentence)` / `stop()` — run the pipeline
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
python -m pytest -q   # 39 tests
```

Tests assert against the **real model assets** (`assets/index.txt` +
`assets/audio/`) rather than hard-coded words/IDs, so they keep passing
(and verify consistency) whether the dictionary grows or shrinks.
