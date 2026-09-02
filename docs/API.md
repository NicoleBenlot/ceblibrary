# ceblibrary API guide

This guide shows every public API and how to call it. It is written for
users embedding the library (Decode text) or cleaning STT output (Corrector),
with runnable examples throughout.

Flow at a glance:

```
user speech -> STT text -> Corrector (normalize/correct) -> Decoder (word -> audio ID) -> <id>.mp3 clips
```

The **repo doubles as the model directory**: copy the whole repo into an app's
`models/` folder (`<app>/models/ceblibrary/`) and point `Decoder` at it.

---

## 1. Install and get a model

```sh
pip install .          # the wheel ships code only — no model data
```

Copy the repo into your app (or fetch it from a release):

```text
my_app/
└── models/
    └── ceblibrary/     <-- the whole repo
        ├── assets/
        │   ├── index.txt   word -> audio ID map
        │   └── audio/      <id>.mp3 clips
        └── ceblibrary/     the Python package
```

---

## 2. Decoder — sentence to audio IDs

### 2.1 Constructing a Decoder

Three ways to point at the model; the right one depends on where the model lives:

```python
from ceblibrary import Decoder

d = Decoder()                          # 1. repo root / package parent (default)
d = Decoder("C:/apps/my_app/models/ceblibrary")   # 2. explicit model directory
d = Decoder(index_path=".../index.txt")           # 3. explicit index file
```

- `Decoder(model=None, index_path=None)` positional `model`, keyword-only `index_path`.
- Model lookup for case 2 checks `<dir>/index.txt` then `<dir>/assets/index.txt`.
- A model that can't be found raises `ModelNotFoundError` (a `FileNotFoundError`).

All lookups are **case-insensitive** (words are lowercased on load).

### 2.2 Decode a sentence

```python
d = Decoder()
d.decode("ako mo open sa balay")
# ['ako','mo','open'...] -> e.g. [24, 30, None, 2, 119-ish]
```

- `decode(sentence) -> List[Optional[int]]` — one audio ID per whitespace-separated word; **unknown words become `None`**.
- `decode_strict(sentence) -> List[int]` — same, but raises `KeyError("Unknown word: ...")` on the first unknown word.

```python
d.decode("pero bisan apan")         # -> [12, 14, 11]   (all known)
d.decode("pero xyz apan")           # -> [12, None, 11]  (xyz unknown)
d.decode_strict("pero xyz apan")    # -> raises KeyError
```

### 2.3 Single-word lookups

```python
d.get_id("apan")      # -> 11         (None if unknown)
d.has_word("pan")     # -> False
d.has_word("PAN")     # -> True        (case-insensitive)
```

### 2.4 Resolve audio files

```python
paths = d.audio_paths("pero bisan apan")   # skips unknown words
# -> [.../assets/audio/12.mp3, .../assets/audio/14.mp3, .../assets/audio/11.mp3]

d.audio_path(12)                      # -> .../assets/audio/12.mp3
# audio_id is REQUIRED; for unknown words nothing is produced

# override the asset dir (e.g. fetch clips elsewhere):
d.audio_paths("pero apan", assets_dir="C:/my/clips")
```

`audio_paths` only returns paths — files may be fetched on demand (clips are
not bundled with the pip package). Every ID in `index.txt` has a matching
`<id>.mp3` in `assets/audio/`.

### 2.5 Model metadata & introspection

```python
d.model_dir           # absolute path of the model dir
d.index_path          # absolute path of the index.txt actually used
d.assets_dir          # model dir + /assets/audio
d.word_count          # number of known words (int)
d.words               # sorted tuple of known words (feeds a Corrector)
d.available_letters() # sorted list of [section] headers, e.g. ["ad", "ak", ...]
d.config              # dict of audio playback settings (see audio_config.json)
```

### 2.6 Building / editing an index

Useful for tooling that grows the dictionary.

```python
d = Decoder()             # load the model
d.add_word("balay", 200)  # register a word -> audio ID (case-insensitive)
d.word_count              # now one larger

d.save()                  # write index.txt (grouped by 2-letter sections)
d.save("path/to/index.txt")   # write to another file
```

`save()` re-writes the index grouped under `[<first two letters>]` headers, so
`Decoder(other_model_dir).add_word(...)` forms a little build pipeline.

---

## 3. Corrector — clean STT / noisy text before decoding

Sits between STT and your AI. Two layers:

1. `normalize()` — vocabulary-free: lowercase, strip diacritics/punctuation,
   collapse repeated letters ("hellooo" -> "hello"), drop fillers (uh, um, ...).
2. `correct()` — normalize + map near-miss tokens to the closest known word by
   edit distance; tokens too far away pass through untouched.

### 3.1 Constructing

```python
from ceblibrary import Corrector, Decoder

# no vocabulary -> normalization only
c = Corrector()

# vocabulary-aware, straight from the model
c = Corrector.from_decoder(Decoder())

# explicit vocabulary (any iterable of strings)
c = Corrector(["pero", "bisan", "apan"])

# control the correction threshold (default: 1 for <=4-letter words, else 2)
c = Corrector.from_decoder(Decoder(), max_distance=2)
```

`Corrector(vocabulary=None, *, max_distance=None)` — `max_distance` is
keyword-only. It is fully **independent of Decoder** unless you opt in.

### 3.2 Normalize (vocabulary-free)

```python
c.normalize("PERO, BISAN!")            # -> "pero bisan"
c.normalize("perooo bisaaaaan")        # -> "pero bisan"     (repeated letters)
c.normalize("peró bisán")              # -> "pero bisan"     (diacritics)
c.normalize("uh pero um bisan")        # -> "pero bisan"     (fillers dropped)
c.normalize("   pero    bisan  ")      # -> "pero bisan"     (whitespace)
```

### 3.3 Correct (vocabulary-aware)

```python
c = Corrector.from_decoder(Decoder())

c.correct("peroo bisan")               # -> "pero bisan"     (typo fixed)
c.correct("PERO, bisa!")               # -> "pero bisan"
c.correct("pero skyscraper bisan")     # -> "pero skyscraper bisan"  (too far -> untouched)
```

Token-level variants return lists instead of joined strings:

```python
c.normalize_tokens("uh AKO, balaay.")  # -> ["ako", "balaay"]
c.correct_tokens("peroo bisa")         # -> ["pero", "bisa"]
```

### 3.4 Candidate helpers

```python
c.is_known("bisan")       # -> True     (case-insensitive)
c.is_known("besan")       # -> False

c.suggest("besan")        # -> ["bisan", ...]  closest words, best first
c.suggest("besan", limit=5)   # more suggestions
```

---

## 4. Full pipeline example

```python
from ceblibrary import Corrector, Decoder

decoder = Decoder("models/ceblibrary")
corrector = Corrector.from_decoder(decoder)

raw_stt = "uh AKO, balaay."
cleaned = corrector.correct(raw_stt)      # -> "ako balay"
ids = decoder.decode(cleaned)             # -> list of audio IDs (None = unknown)
paths = decoder.audio_paths(cleaned)      # -> paths to <id>.mp3 for known words

print(ids)     # e.g. [24, 118]
```

Send `paths` (or stream the decoded IDs) to your audio player.

---

## 5. Audio config (`audio_config.json`)

If you drop an `audio_config.json` next to `index.txt`, `Decoder.config`
exposes it for your playback/concatenation logic:

```json
{
  "crossfade_ms": 350,
  "fade_in_ms": 180,
  "fade_out_ms": 150,
  "strip_silence_threshold_dbfs": -35,
  "strip_silence_padding_ms": 5,
  "normalize_target_dbfs": -18.0
}
```

With no file, these defaults are used. `demo.py` shows a full
load → trim → normalize → crossfade → play implementation.

---

## 6. Index file format (if you build models)

```
[pa]
pero = 12
[bi]
bisan = 14
```

- Section headers `[xx]` = first two letters of the words inside (lookup skips unrelated prefixes).
- `word = id` with spaces around `=`.
- `#` comments and blank lines ignored; duplicate words: last entry wins.
- IDs are ints (non-int entries ignored); words lowercased on load.

---

## 7. Common errors, explained

| Symptom | Cause | Fix |
|---|---|---|
| `ModelNotFoundError` | model dir has no `index.txt` / `assets/index.txt` | pass the right path or `index_path=` |
| `KeyError: "Unknown word: 'xyz'"` | `decode_strict` hit a word not in the index | use `decode()`, or correct text first |
| `None` entries in `decode()` | unknown words in the sentence | run `Corrector.from_decoder(...).correct(...)` first |
| empty `audio_paths` | no known words (or all clips missing) | check `has_word` / `assets/audio/` |

---

## 8. Index of all public callables

| Call | Returns |
|---|---|
| `Decoder(model=None, *, index_path=None)` | a `Decoder` |
| `Decoder.decode(sentence)` | `list[int \| None]` |
| `Decoder.decode_strict(sentence)` | `list[int]` (raises `KeyError`) |
| `Decoder.get_id(word)` | `int \| None` |
| `Decoder.has_word(word)` | `bool` |
| `Decoder.audio_path(id, assets_dir=None)` | `str` |
| `Decoder.audio_paths(sentence, assets_dir=None)` | `list[str]` |
| `Decoder.add_word(word, id)` | `None` |
| `Decoder.save(path=None)` | `None` |
| `Decoder.model_dir / index_path / assets_dir / config` | `str / str / str / dict` |
| `Decoder.word_count / words / available_letters()` | `int / tuple / list` |
| `Corrector(vocab=None, *, max_distance=None)` | a `Corrector` |
| `Corrector.from_decoder(decoder)` | a `Corrector` |
| `Corrector.normalize(text) / normalize_tokens(text)` | `str / list[str]` |
| `Corrector.correct(text) / correct_tokens(text)` | `str / list[str]` |
| `Corrector.suggest(word, limit=3)` | `list[str]` |
| `Corrector.is_known(word)` | `bool` |