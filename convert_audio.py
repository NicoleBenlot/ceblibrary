"""Convert the model's MP3 word clips to smaller Opus clips.

Run once with ffmpeg available. Converts every `assets/audio/*.mp3` to
`assets/audio/*.opus` at a speech-transparent bitrate (48 kbps), then (with
--delete-mp3) removes the originals.

Opus is roughly half the size of MP3 at the same perceived speech quality,
shrinking the embeddable model. After conversion, point the Decoder/StreamPlayer
at the opus files by setting the model's `audio_format` to "opus" in
audio_config.json (or pass it via a Decoder subclass/config).

Requires ffmpeg on PATH:

    ffmpeg -version

Usage:

    python convert_audio.py                 # convert, keep mp3s
    python convert_audio.py --delete-mp3    # convert and remove mp3s
    python convert_audio.py --bitrate 64k   # override bitrate
"""

import argparse
import glob
import os
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert assets/audio/*.mp3 to *.opus (needs ffmpeg)."
    )
    parser.add_argument(
        "--audio-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "audio"),
        help="directory holding the word clips (default: assets/audio)",
    )
    parser.add_argument(
        "--bitrate",
        default="48k",
        help="Opus bitrate, e.g. 32k, 48k (default 48k)",
    )
    parser.add_argument(
        "--delete-mp3",
        action="store_true",
        help="delete the source .mp3 files after a successful conversion",
    )
    args = parser.parse_args(argv)

    audio_dir = args.audio_dir
    if not os.path.isdir(audio_dir):
        raise SystemExit(f"audio dir not found: {audio_dir}")

    mp3s = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    if not mp3s:
        print(f"no .mp3 files found in {audio_dir}")
        return

    for idx, src in enumerate(mp3s, 1):
        dst = os.path.splitext(src)[0] + ".opus"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", src,
            "-c:a", "libopus", "-b:a", args.bitrate,
            dst,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[{idx}/{len(mp3s)}] FAILED {os.path.basename(src)}: {exc}")
            continue
        except FileNotFoundError:
            raise SystemExit(
                "ffmpeg not found on PATH. Install ffmpeg first "
                "(choco install ffmpeg / winget install ffmpeg / ffmpeg.org)."
            )
        before = os.path.getsize(src)
        after = os.path.getsize(dst)
        if args.delete_mp3:
            os.remove(src)
        print(
            f"[{idx}/{len(mp3s)}] {os.path.basename(src)} -> "
            f"{os.path.basename(dst)}  ({before} -> {after} bytes; "
            f"{-100 * (1 - after / before):.0f}%)"
        )

    print("done.")


if __name__ == "__main__":
    main()
