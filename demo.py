"""Demo: hear the ceblibrary model with the real index.txt + assets/audio.

Builds an audio sentence from the model's word clips, concatenates them into
one MP3/WAV, writes it out, and optionally plays it through the speakers.

Uses the real model at the repo root via Decoder() (no hardcoded vocab);
the dictionary is assets/index.txt with audio in assets/audio/.

Examples:
    python demo.py "ako mo open sa balay"
    python demo.py --play "ako kan nimo"
    python demo.py "ako balay" -o out.mp3
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from ceblibrary import Corrector, Decoder, StreamPlayer


def _normalize(clip, cfg):
    """Normalize a clip to a consistent target loudness."""
    change = cfg["normalize_target_dbfs"] - clip.dBFS
    return clip.apply_gain(change)


def _trim(clip, cfg):
    """Remove leading/trailing silence."""
    return clip.strip_silence(
        silence_thresh=cfg["strip_silence_threshold_dbfs"],
        padding=cfg["strip_silence_padding_ms"],
    )


def build_audio(decoder, sentence):
    """Concatenate model audio clips with crossfades and normalized volume."""
    cfg = decoder.config

    crossfade_ms = cfg["crossfade_ms"]
    fade_in_ms = cfg["fade_in_ms"]
    fade_out_ms = cfg["fade_out_ms"]

    clips = []
    for word in sentence.split():
        audio_id = decoder.get_id(word)
        if audio_id is None:
            print(f"  [skip] unknown word: {word!r}")
            continue
        path = decoder.audio_path(audio_id)
        if not os.path.isfile(path):
            print(f"  [skip] missing clip: {path}")
            continue
        clips.append(_normalize(_trim(AudioSegment.from_file(path), cfg), cfg))
        print(f"  {word:8} -> {audio_id}  from {os.path.basename(path)}")

    if not clips:
        raise SystemExit("no known words to play; nothing built")

    combined = clips[0].fade_in(fade_in_ms).fade_out(fade_out_ms)
    for clip in clips[1:]:
        combined = combined.append(clip.fade_in(fade_in_ms).fade_out(fade_out_ms), crossfade=crossfade_ms)
    return combined


def play(segment):
    """Play AudioSegment through the default output device."""
    samples = np.array(segment.get_array_of_samples())
    if segment.channels > 1:
        samples = samples.reshape(-1, segment.channels)
    sd.play(samples, segment.frame_rate)
    sd.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Demo the ceblibrary model: decode + play a sentence."
    )
    parser.add_argument("sentence", help="Cebuano sentence to speak")
    parser.add_argument("-o", "--out", help="write concatenated audio file (.mp3/.wav)")
    parser.add_argument(
        "--play",
        action="store_true",
        help="play through the speakers (requires an audio device)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="use streaming/pipelined playback (lower first-audio latency)",
    )
    parser.add_argument(
        "--model", default=None, help="optional model dir (defaults to repo root)"
    )
    args = parser.parse_args(argv)

    decoder = Decoder(args.model)
    corrector = Corrector.from_decoder(decoder)
    sentence = corrector.correct(args.sentence)
    print(f"model  : {decoder.model_dir}")
    print(f"words  : {decoder.word_count}")
    print(f"audio  : {decoder.assets_dir}")
    print(f"input   : {args.sentence!r}")
    print(f"cleaned: {sentence!r}")
    print("decoded audio ids:", decoder.decode(sentence))

    if args.stream:
        player = StreamPlayer(decoder)
        print("streaming playback...")
        player.play(sentence)
        print(f"loaded {player.loaded_count} clips, skipped {player.failed_count}")
        played = True
        return

    print("building audio...")

    audio = build_audio(decoder, sentence)

    played = False
    if args.play:
        print("playing...")
        play(audio)
        played = True

    if args.out:
        audio.export(args.out, format="mp3" if args.out.lower().endswith(".mp3") else "wav")
        print(f"wrote  : {os.path.abspath(args.out)}")

    if not args.play and not args.out:
        # Default demo: play it so the user actually hears it.
        print("playing...")
        play(audio)
        played = True

    print("done.")


if __name__ == "__main__":
    main()
