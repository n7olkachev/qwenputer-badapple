"""Measure how much Bad Apple!! compresses at various sizes, to pick what fits in the 128 KB DROM.

usage: uv run python tools/measure.py video/badapple.mp4
"""
import subprocess
import sys
import zlib

import numpy as np


def frames(path, w, h, fps):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={w}:{h}:flags=area,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
    return a >= 128  # bool [N, H, W]


def pack(bits):  # [N, H, W] bool -> [N, H*W/8] bytes, MSB first
    return np.packbits(bits, axis=-1).reshape(bits.shape[0], -1)


def rle_zero_lit(data: bytes) -> int:
    """Size of a byte-RLE: control c < 128 -> c+1 literals follow; c >= 128 -> c-127 zero bytes."""
    out = 0
    i, n = 0, len(data)
    while i < n:
        if data[i] == 0:
            j = i
            while j < n and data[j] == 0 and j - i < 128:
                j += 1
            out += 1
            i = j
        else:
            j = i
            while j < n and data[j] != 0 and j - i < 128:
                j += 1
            out += 1 + (j - i)
            i = j
    return out


def lz_size(data: bytes, window=4096, minlen=3) -> int:
    """Greedy LZ77 with 2-byte match tokens (12-bit offset, 4-bit length-3) and literal runs.
    Rough size estimate only; the real encoder may do better with optimal parsing."""
    n = len(data)
    i = 0
    out = 0
    lit = 0
    # hash chain
    head = {}
    while i < n:
        best_len, best_off = 0, 0
        if i + minlen <= n:
            key = data[i:i + minlen]
            for j in head.get(key, [])[-32:][::-1]:
                if i - j > window:
                    break
                l = 0
                while i + l < n and l < 18 and data[j + l] == data[i + l]:
                    l += 1
                if l > best_len:
                    best_len, best_off = l, i - j
        if best_len >= minlen:
            if lit:
                out += 1 + lit
                lit = 0
            out += 2
            for k in range(best_len):
                if i + k + minlen <= n:
                    head.setdefault(data[i + k:i + k + minlen], []).append(i + k)
            i += best_len
        else:
            lit += 1
            if lit == 128:
                out += 1 + lit
                lit = 0
            if i + minlen <= n:
                head.setdefault(data[i:i + minlen], []).append(i)
            i += 1
    if lit:
        out += 1 + lit
    return out


def main():
    path = sys.argv[1]
    configs = [(64, 48, 15), (64, 48, 12), (64, 48, 10), (80, 60, 12), (80, 60, 10),
               (96, 72, 12), (96, 72, 10), (128, 96, 10), (128, 96, 8)]
    print(f"{'size':>9} {'fps':>3} {'frames':>6} {'raw KB':>7} {'zlib':>6} {'zlib-xor':>8} "
          f"{'rle-xor':>7} {'lz-xor':>7}")
    for w, h, fps in configs:
        f = frames(path, w, h, fps)
        p = pack(f)
        raw = p.tobytes()
        xor = np.concatenate([p[:1], p[1:] ^ p[:-1]]).tobytes()
        kb = lambda b: f"{b / 1024:6.0f}K"
        print(f"{w:>4}x{h:<4} {fps:>3} {len(f):>6} {kb(len(raw)):>7} {kb(len(zlib.compress(raw, 9))):>6} "
              f"{kb(len(zlib.compress(xor, 9))):>8} {kb(rle_zero_lit(xor)):>7} {kb(lz_size(xor)):>7}",
              flush=True)


if __name__ == "__main__":
    main()
