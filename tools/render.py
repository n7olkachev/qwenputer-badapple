"""Render the bytes the machine printed (build/out.bin, packed 1-bit rows) into a video with the
original audio.

usage: python tools/render.py [--in build/out.bin] [--out build/badapple_qwenputer.mp4] [--scale 8]
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def config():
    cfg = dict(line.split()[1:3] for line in (ROOT / "build/config.h").read_text().splitlines())
    return int(cfg["FRAME_W"]), int(cfg["FRAME_H"]), int(cfg["FPS"])


def unpack(data: bytes, w: int, h: int) -> np.ndarray:
    fb = w * h // 8
    n = len(data) // fb
    bits = np.unpackbits(np.frombuffer(data[:n * fb], np.uint8)).reshape(n, h, w)
    return bits * 255  # [N, H, W] uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "build/out.bin"))
    ap.add_argument("--out", default=str(ROOT / "build/badapple_qwenputer.mp4"))
    ap.add_argument("--audio", default=str(ROOT / "video/badapple.mp4"))
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--gif", action="store_true", help="also write a GIF of the first 10 s")
    a = ap.parse_args()
    w, h, fps = config()
    frames = unpack(Path(a.inp).read_bytes(), w, h)
    print(f"{len(frames)} frames {w}x{h} @ {fps} fps = {len(frames) / fps:.1f} s")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-i", a.audio, "-map", "0:v", "-map", "1:a?",
           "-vf", f"scale={w * a.scale}:{h * a.scale}:flags=neighbor",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", a.out]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    print("wrote", a.out)
    if a.gif:
        gif = str(Path(a.out).with_suffix(".gif"))
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "-", "-vf", f"scale={w * 4}:{h * 4}:flags=neighbor", gif]
        subprocess.run(cmd, input=frames[:fps * 10].tobytes(), check=True)
        print("wrote", gif)


if __name__ == "__main__":
    main()
