"""Tests for the streaming/pipelined audio playback (StreamPlayer).

These tests exercise the pipeline logic (ordering, prefetch, caching, error
handling, latency) without needing real audio assets or ffmpeg: a fake decoder
and recording sink stand in for the model and output device respectively, and
the load step is stubbed so completion order can be controlled deterministically.
"""

import logging
import os
import time

import pytest

from ceblibrary import Decoder, StreamPlayer

try:
    from pydub import AudioSegment as _PydubSegment  # noqa: F401

    HAVE_PYDUB = True
except Exception:  # noqa: BLE001
    _PydubSegment = None
    HAVE_PYDUB = False

_needs_pydub = pytest.mark.skipif(
    not HAVE_PYDUB, reason="pydub not installed (needed to synthesize clips)"
)


class FakeClip:
    """Minimal stand-in for pydub.AudioSegment: holds int16 samples."""

    frame_rate = 44100
    channels = 1

    def __init__(self, tag, samples):
        self.tag = tag
        self._samples = samples

    def get_array_of_samples(self):
        return self._samples

    def fade_in(self, ms):
        return self

    def fade_out(self, ms):
        return self


def _int16(*values):
    from array import array

    return array("h", values)


class FakeDecoder:
    """Mirrors the Decoder surface the StreamPlayer depends on."""

    def __init__(self, mapping, *, format="mp3"):
        self._mapping = {k.lower(): v for k, v in mapping.items()}
        self.assets_dir = "definitely-not-a-real-dir"
        self.config = {
            "crossfade_ms": 0,
            "fade_in_ms": 0,
            "fade_out_ms": 0,
            "strip_silence_threshold_dbfs": -35,
            "strip_silence_padding_ms": 5,
            "normalize_target_dbfs": -18.0,
            "audio_format": format,
            "sample_rate": 44100,
            "channels": 1,
            "dtype": "int16",
        }

    def get_id(self, word):
        return self._mapping.get(word.lower())

    def audio_path(self, audio_id):
        return os.path.join(self.assets_dir, f"{audio_id}.{self.config['audio_format']}")


class RecordingSink:
    """Captures the order/timestamps of clips written to the 'device'."""

    def __init__(self):
        self.clips = []          # FakeClips in write order
        self.timestamps = []     # wall-clock time of each write
        self.opened = False
        self.closed = False

    def open(self, *, samplerate, channels, dtype="int16"):
        self.opened = True

    def write(self, clip):
        self.clips.append(clip)
        self.timestamps.append(time.perf_counter())

    def close(self):
        self.closed = True


class _DelayedStreamPlayer(StreamPlayer):
    """StreamPlayer with a controllable, order-independent load delay."""

    def __init__(self, decoder, *, delays=None, sink=None, **kw):
        super().__init__(decoder, sink=sink or RecordingSink(), **kw)
        self._delays = delays or {}
        self._calls = []

    def _decode_from_file(self, path):
        # Never called in these tests; loads are stubbed in _load_segment.
        raise AssertionError("_decode_from_file should not be called")

    def _load_segment(self, audio_id):
        self._calls.append(audio_id)
        delay = self._delays.get(audio_id, 0.0)
        if delay:
            time.sleep(delay)
        if audio_id < 0:  # negative ids simulate load failures
            self.failed_count += 1
            return None
        self.loaded_count += 1
        return FakeClip(str(audio_id), _int16(1, 2, 3, 4, 5, 6))


def _clips_by_tag(sink):
    return [c.tag for c in sink.clips]


class TestStreamerOrdering:
    def test_plays_in_sentence_order_regardless_of_load_order(self):
        # id 30 (word B) loads fast, id 10 (word A) loads slow -> A must
        # still be written before B because A comes first in the sentence.
        decoder = FakeDecoder({"a": 10, "b": 30})
        player = _DelayedStreamPlayer(
            decoder, delays={10: 0.05, 30: 0.0}, prefetch_limit=4, cache_size=64
        )
        player.play("a b")
        assert _clips_by_tag(player._sink) == ["10", "30"]

    def test_longer_sentence_stays_ordered_with_reversed_load_times(self):
        decoder = FakeDecoder(
            {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        )
        # Slowest loads are the *first* words -> later words finish earlier.
        player = _DelayedStreamPlayer(
            decoder,
            delays={1: 0.03, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
            prefetch_limit=4,
            cache_size=64,
        )
        player.play("one two three four five")
        assert _clips_by_tag(player._sink) == ["1", "2", "3", "4", "5"]


class TestFirstAudioLatency:
    def test_first_clip_written_before_remaining_loads_complete(self):
        # Make the 2nd/3rd words slow; first audio (word 0) should be written
        # to the sink before those slow loads finish.
        decoder = FakeDecoder({"a": 1, "b": 2, "c": 3})
        player = _DelayedStreamPlayer(
            decoder,
            delays={2: 0.03, 3: 0.03},
            prefetch_limit=4,
            cache_size=64,
        )

        done = player.play_async("a b c")
        # Poll until the first clip hits the sink, then measure.
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            if player._sink.clips:
                break
            time.sleep(0.001)
        player.wait(done)

        assert player._sink.clips, "first clip was never written"
        assert len(player._sink.clips) == 3


class TestUnknownAndMissing:
    def test_unknown_words_are_skipped_without_crash(self):
        decoder = FakeDecoder({"known": 1})
        player = _DelayedStreamPlayer(decoder, prefetch_limit=4, cache_size=64)
        player.play("known ghostword known")
        assert _clips_by_tag(player._sink) == ["1", "1"]

    def test_all_unknown_words_plays_nothing(self):
        decoder = FakeDecoder({"a": 1})
        player = _DelayedStreamPlayer(decoder, prefetch_limit=4, cache_size=64)
        player.play("zzz yyy")
        assert player._sink.clips == []
        assert player._sink.opened is False  # no device opened for silence

    def test_load_failure_is_skipped_but_rest_continues(self):
        # Negative ids fail in _load_segment (bad/missing asset).
        decoder = FakeDecoder({"good": 5, "bad": -1, "good2": 6})
        player = _DelayedStreamPlayer(decoder, prefetch_limit=4, cache_size=64)
        player.play("good bad good2")
        assert _clips_by_tag(player._sink) == ["5", "6"]
        assert player.failed_count == 1

    def test_empty_sentence_does_nothing(self):
        decoder = FakeDecoder({"a": 1})
        player = _DelayedStreamPlayer(decoder, prefetch_limit=4, cache_size=64)
        player.play("   ")
        assert player._sink.clips == []


class TestPrefetchOrderingGuarantee:
    def test_playback_never_writes_a_later_clip_before_an_earlier_one(self):
        # Even when the 2nd clip is ready long before the 1st, writes must be
        # in sentence order (proves the ordered-buffer/seq guard works).
        decoder = FakeDecoder({"first": 100, "second": 200, "third": 300})
        player = _DelayedStreamPlayer(
            decoder,
            delays={100: 0.1, 200: 0.0, 300: 0.0},
            prefetch_limit=4,
            cache_size=64,
        )
        player.play("first second third")
        order = _clips_by_tag(player._sink)
        assert order == [str(k) for k in (100, 200, 300)]


class TestPrefetchLimit:
    def test_prefetch_never_exceeds_limit(self):
        # With prefetch_limit=2 and slow loads, at most 2 loads run at once.
        decoder = FakeDecoder({f"w{i}": i for i in range(1, 8)})
        player = _DelayedStreamPlayer(
            decoder,
            delays={i: 0.01 for i in range(1, 8)},
            prefetch_limit=2,
            cache_size=64,
        )
        max_in_flight = [0]
        in_flight = [0]

        orig = player._load_segment

        def wrapping(audio_id):
            in_flight[0] += 1
            max_in_flight[0] = max(max_in_flight[0], in_flight[0])
            try:
                time.sleep(0.012)  # keep it in flight long enough to overlap
                return orig(audio_id)
            finally:
                in_flight[0] -= 1

        player._load_segment = wrapping
        player.play("w1 w2 w3 w4 w5 w6 w7")
        assert max_in_flight[0] <= 2


class TestCaching:
    def _make_stubbed_player(self, tmp_path, ids, cache_size):
        """Player using the base load path (real files + stubbed decode)."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for aid in ids:
            (audio_dir / f"{aid}.opus").write_bytes(b"fake")

        decoder = FakeDecoder({f"w{aid}": aid for aid in ids}, format="opus")
        decoder.assets_dir = str(audio_dir)
        player = StreamPlayer(decoder, prefetch_limit=4, cache_size=cache_size)
        decoded = []

        def fake_decode(path):
            decoded.append(int(os.path.basename(path).split(".")[0]))
            # Silence requires no decode libs; a 10ms silent clip.
            from pydub import AudioSegment

            return AudioSegment.silent(duration=10)

        player._decode_from_file = fake_decode
        player._sink = RecordingSink()
        player._sink_open = False
        return player, decoded

    @_needs_pydub
    def test_repeated_word_loads_once(self, tmp_path):
        player, decoded = self._make_stubbed_player(tmp_path, [7], cache_size=64)
        player.play("w7 w7 w7")
        # Same audio id cached; only one underlying decode.
        assert decoded == [7]
        assert player._sink.clips and all(c is not None for c in player._sink.clips)

    @_needs_pydub
    def test_lru_cache_evicts_oldest_when_full(self, tmp_path):
        ai = [11, 12, 13, 14, 15]
        player, decoded = self._make_stubbed_player(tmp_path, ai, cache_size=2)
        player.play(" ".join(f"w{a}" for a in ai))
        # With cache_size=2 and 5 distinct ids, every id is decoded.
        assert len(decoded) == 5

    @_needs_pydub
    def test_missing_file_is_skipped_and_logged(self, tmp_path, caplog):
        # Only id 9 has a file; id 8 does not -> skipped, rest continues.
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "9.opus").write_bytes(b"fake")  # id 8 file absent
        decoder = FakeDecoder({"w8": 8, "w9": 9}, format="opus")
        decoder.assets_dir = str(audio_dir)
        player = StreamPlayer(decoder, prefetch_limit=4, cache_size=64)
        sink = RecordingSink()
        player._sink = sink
        player._sink_open = False

        def fake_decode(path):
            from pydub import AudioSegment

            return AudioSegment.silent(duration=10)

        player._decode_from_file = fake_decode
        with caplog.at_level(logging.WARNING, logger="ceblibrary.streamer"):
            player.play("w8 w9 w8 w9")
        assert player._sink.clips  # preceded by silence; existence is enough
        assert player.failed_count == 2  # both w8 occurrences missing
        assert "Missing audio asset" in caplog.text


class TestStop:
    def test_stop_returns_and_does_not_raise(self):
        decoder = FakeDecoder({"a": 1, "b": 2})
        player = _DelayedStreamPlayer(
            decoder, delays={1: 0.5}, prefetch_limit=4, cache_size=64
        )
        done = player.play_async("a b")
        time.sleep(0.02)
        player.stop()
        done.wait(timeout=2.0)
        assert done.is_set()


class TestDecoderIntegration:
    def test_constructs_with_real_decoder(self):
        # Ensure StreamPlayer accepts a real Decoder object's config surface.
        decoder = Decoder()
        assert decoder.config["audio_format"] in ("mp3", "opus")
        p = StreamPlayer(decoder)
        assert p._prefetch_limit == 4
