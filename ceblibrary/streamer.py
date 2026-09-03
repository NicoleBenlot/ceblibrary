"""Streaming / pipelined audio playback.

The Decoder maps words to audio IDs but leaves loading and playback to the
caller, and the naive approach is synchronous: evaluate the whole sentence,
fetch every clip, join them, then start playback. For long sentences that is
a lot of latency before anything is heard.

`StreamPlayer` replaces that with a producer/consumer pipeline::

    sentence
      |
      v
  producer thread  -- decode word -> id (incremental)
      |
      v
  ThreadPoolExecutor -- async load (bounded prefetch, LRU cache)
      |
      v
  ordered buffer (seq-indexed, finalized in sentence order)
      |
      v
  playback feed  -- push each ready clip to the audio sink as it arrives

Audio is always consumed in sentence order (guaranteed by sequence numbers),
even when a later clip finishes loading before an earlier one. First audio
starts as soon as the first word's clip is ready; remaining words are
evaluated and prefetched while playback runs. The ordering guarantee holds
because the playback feed never writes a clip until the one before it has
been written.
"""

import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("ceblibrary.streamer")

_UNKNOWN = object()  # sentinel: word had no audio id (skipped)
_FAILED = object()  # sentinel: clip could not be loaded (skipped)


class StreamPlayer:
    """Stream + play a sentence with pipelined, ordered audio.

    Parameters
    ----------
    decoder : Decoder
        Word -> audio ID mapping plus audio config (crossfade, fades).
    prefetch_limit : int
        Max number of clips loading concurrently (the prefetch buffer depth).
        Defaults to 4.
    cache_size : int
        Max number of recently used audio segments kept in memory so repeated
        words don't reload from disk. Defaults to 64.
    sink : object, optional
        The incremental playback sink. Must provide :meth:`open` (with
        frame_rate/channels/dtype), :meth:`write` (append an AudioSegment-like
        clip), and :meth:`close` (flush + stop). Defaults to
        :class:`SounddeviceSink`.
    prefetch_callbacks : callable, optional
        Called with ``(seq_key, word)`` when a word is evaluated and
        ``(seq_key, loaded_or_None)`` when its clip finishes loading. Used by
        tests to observe pipeline progression without real audio.

    Attributes
    ----------
    loaded_count : int
        Total number of clips actually loaded from disk (cache misses).
    failed_count : int
        Total number of clips that could not be loaded (missing/unreadable).
    """

    def __init__(
        self,
        decoder,
        *,
        prefetch_limit: int = 4,
        cache_size: int = 64,
        sink=None,
        prefetch_callbacks=None,
    ):
        if prefetch_limit < 1:
            raise ValueError("prefetch_limit must be >= 1")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")

        self._decoder = decoder
        self._prefetch_limit = prefetch_limit
        self._cache_size = cache_size
        self._cfg = decoder.config
        self._sink = sink if sink is not None else SounddeviceSink()
        self._prefetch_callbacks = prefetch_callbacks

        self._cache: "OrderedDict[int, object]" = OrderedDict()
        self._sem = threading.Semaphore(prefetch_limit)
        self._cond = threading.Condition()
        self._ordered: Dict[int, object] = {}
        self._next_to_play = 0
        self._stopped = False
        self._executor: Optional[ThreadPoolExecutor] = None

        self.loaded_count = 0
        self.failed_count = 0
        self._sink_open = False

    # -- public playback ---------------------------------------------------

    def play(self, sentence: str) -> None:
        """Blocking: evaluate, prefetch, and play the sentence.

        Returns when playback completes (or is stopped). Unknown words and
        missing/unreadable clips are skipped without crashing.
        """
        self.play_async(sentence).wait()

    def play_async(self, sentence: str) -> threading.Event:
        """Kick off playback; returns a threading.Event set when done.

        A producer and a playback worker run concurrently, so first audio
        begins as soon as the first clip is ready while later clips load.
        """
        done = threading.Event()
        self._stopped = False
        self._ordered.clear()
        self._next_to_play = 0
        self._sink_open = False
        self._executor = ThreadPoolExecutor(
            max_workers=self._prefetch_limit,
            thread_name_prefix="cebl-prefetch",
        )
        threading.Thread(
            target=self._run,
            args=(sentence, done),
            name="cebl-pipeline",
            daemon=True,
        ).start()
        return done

    def stop(self) -> None:
        """Request cancellation of in-flight playback and drain the workers."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def wait(self, done: Optional[threading.Event] = None) -> None:
        """Block until the given (or, if none, any in-flight) playback ends."""
        if done is not None:
            done.wait()

    # -- pipeline orchestration --------------------------------------------

    def _run(self, sentence: str, done: threading.Event) -> None:
        try:
            words = sentence.split()
            total = len(words)

            producer = threading.Thread(
                target=self._produce,
                args=(words,),
                name="cebl-evaluate",
                daemon=True,
            )
            consumer = threading.Thread(
                target=self._consume,
                args=(total,),
                name="cebl-playback",
                daemon=True,
            )
            producer.start()
            consumer.start()
            producer.join()
            consumer.join()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            done.set()

    def _produce(self, words) -> None:
        """Evaluate words incrementally and submit bounded prefetch loads."""
        for seq, word in enumerate(words):
            if self._stopped:
                return
            audio_id = self._decoder.get_id(word)
            if self._prefetch_callbacks is not None:
                self._prefetch_callbacks("eval", (seq, word, audio_id))
            if audio_id is None:
                # Unknown word: finalize slot as skipped immediately so the
                # consumer doesn't stall waiting for a load that never runs.
                self._finalize(seq, _UNKNOWN)
                continue

            self._sem.acquire()
            if self._stopped:
                self._sem.release()
                return
            self._executor.submit(self._load_and_finalize, seq, audio_id)

    def _load_and_finalize(self, seq: int, audio_id: int) -> None:
        try:
            segment = self._load_segment(audio_id)
            if self._prefetch_callbacks is not None:
                self._prefetch_callbacks("load", (seq, audio_id, segment))
            self._finalize(seq, segment if segment is not None else _FAILED)
        finally:
            self._sem.release()

    def _finalize(self, seq: int, value: object) -> None:
        with self._cond:
            if self._stopped:
                return
            self._ordered[seq] = value
            self._cond.notify_all()

    def _consume(self, total: int) -> None:
        fade_in = int(self._cfg.get("fade_in_ms", 0))
        fade_out = int(self._cfg.get("fade_out_ms", 0))
        crossfade = int(self._cfg.get("crossfade_ms", 0))
        frame_rate = int(self._cfg.get("sample_rate", 48000))
        channels = int(self._cfg.get("channels", 1))
        dtype = str(self._cfg.get("dtype", "int16"))

        played_any = False
        while self._next_to_play < total:
            if self._stopped:
                break
            seq = self._next_to_play
            value = self._wait_for(seq)
            if self._stopped:
                break

            if value is not _UNKNOWN and value is not _FAILED:
                try:
                    self._ensure_sink(frame_rate, channels, dtype)
                    self._sink.write(self._trimmed(value, fade_in, fade_out, seq))
                    played_any = True
                except Exception as exc:  # noqa: BLE001 - skip bad clip
                    logger.error("Failed to stream clip %d: %s", seq, exc)

            self._next_to_play += 1

        try:
            self._close_sink()
        finally:
            if not played_any and self._sink_open:
                logger.info("Nothing playable was streamed")

    def _trimmed(self, segment, fade_in, fade_out, seq):
        try:
            clip = segment.fade_in(fade_in).fade_out(fade_out)
            return clip
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fades skipped for clip %d: %s", seq, exc)
            return segment

    def _ensure_sink(self, frame_rate: int, channels: int, dtype: str) -> None:
        if not self._sink_open:
            self._sink.open(
                samplerate=frame_rate, channels=channels, dtype=dtype
            )
            self._sink_open = True

    def _close_sink(self) -> None:
        if self._sink_open:
            try:
                self._sink.close()
            finally:
                self._sink_open = False

    def _wait_for(self, seq: int) -> object:
        with self._cond:
            while seq not in self._ordered and not self._stopped:
                self._cond.wait(timeout=0.05)
            return self._ordered.get(seq)

    # -- loading / caching -------------------------------------------------

    def _load_segment(self, audio_id: int):
        """Return the decoded clip for an audio id (cached), or None on failure."""
        cached = self._cache_get(audio_id)
        if cached is not None:
            return cached

        path = self._decoder.audio_path(audio_id)
        if not os.path.isfile(path):
            logger.warning("Missing audio asset skipped: %s (id=%d)", path, audio_id)
            self.failed_count += 1
            return None

        try:
            segment = self._decode_from_file(path)
            if segment.sample_width != 2:
                segment = segment.set_sample_width(2)
            self.loaded_count += 1
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.error("Failed to load audio %s (id=%d): %s", path, audio_id, exc)
            self.failed_count += 1
            return None

        segment = self._apply_processing(segment)
        self._cache_put(audio_id, segment)
        return segment

    def _decode_from_file(self, path: str):
        """Load an audio file into a clip. pydub imported lazily (optional)."""
        from pydub import AudioSegment

        return AudioSegment.from_file(path)

    def _apply_processing(self, segment):
        """Strip silence and normalize loudness per audio_config.json."""
        cfg = self._cfg
        try:
            segment = segment.strip_silence(
                silence_thresh=cfg["strip_silence_threshold_dbfs"],
                padding=cfg["strip_silence_padding_ms"],
            )
        except Exception as exc:  # noqa: BLE001 - trim is best-effort
            logger.debug("Silence-strip skipped: %s", exc)

        try:
            segment = segment.apply_gain(cfg["normalize_target_dbfs"] - segment.dBFS)
        except Exception as exc:  # noqa: BLE001 - normalize is best-effort
            logger.debug("Normalize skipped: %s", exc)
        return segment

    def _cache_get(self, audio_id: int):
        with self._cond:
            if audio_id in self._cache:
                self._cache.move_to_end(audio_id)
                return self._cache[audio_id]
        return None

    def _cache_put(self, audio_id: int, segment) -> None:
        with self._cond:
            if audio_id in self._cache:
                self._cache.move_to_end(audio_id)
            self._cache[audio_id] = segment
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)


class SounddeviceSink:
    """Incremental playback sink that streams clips to a sounddevice output.

    Each :meth:`write` appends an AudioSegment-like clip to an internal feed
    buffer, and the buffer is drained to an ``sd.OutputStream`` so the device
    starts sounding as soon as the first clip arrives instead of after the
    whole sentence is ready. Clips are pushed back-to-back (no gaps); a clip is
    never written until the one that precedes it has been written, which is
    how playback order is preserved.
    """

    def __init__(self, output_stream=None):
        self._stream = output_stream
        self._buffer = b""
        self._frame_rate = None
        self._channels = 1

    def open(self, *, samplerate, channels, dtype="int16") -> None:
        if self._stream is None:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
            )
        self._stream.start()
        self._frame_rate = samplerate
        self._channels = max(1, channels)

    def write(self, clip) -> None:
        """Append a clip (AudioSegment-like) to the playout buffer."""
        if self._stream is None or self._frame_rate is None:
            raise RuntimeError("sink not opened")
        self._buffer += clip.get_array_of_samples().tobytes()
        # Drain incrementally so the device starts sounding early; the OS
        # audio buffer smooths the chunk boundaries into the stream.
        bytes_per_frame = self._channels * 2
        chunk = max(1, bytes_per_frame * 1024)  # ~1024 frames per drain
        nbytes = (len(self._buffer) // chunk) * chunk
        if nbytes:
            self._stream.write(
                np.frombuffer(self._buffer[:nbytes], dtype=np.int16)
            )
            self._buffer = self._buffer[nbytes:]

    def close(self) -> None:
        if self._stream is not None:
            if self._buffer:
                self._stream.write(
                    np.frombuffer(self._buffer, dtype=np.int16)
                )
                self._buffer = b""
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._frame_rate = None
