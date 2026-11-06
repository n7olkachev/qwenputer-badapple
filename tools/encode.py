"""Encode Bad Apple!! for the guest: video -> 1-bit frames -> ZX0 -> DROM blob + ROM lui-sled.

usage: python tools/encode.py video/badapple.mp4 --size 80x60 --fps 12 [--max-seconds N]

Writes build/raw.bin (the packed frames the guest must print), build/video.zx0, build/drom.bin,
build/sled.S and build/config.h.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from measure import frames, pack  # noqa: E402

ROOT = Path(__file__).parent.parent
ZX0 = ROOT / "third_party/ZX0/src/zx0_6656"
DROM_MAX = 128 * 1024
SLED_REGS = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
             "s1"]
BLOCK_DATA = 40  # 16 x 20 bits
BLOCK_ROM = 17 * 4  # 16 lui + ret


def to_imm20(block: bytes) -> list[int]:
    """40 bytes -> 16 x 20-bit values (each 5 bytes -> 2 values, little-endian bit order)."""
    assert len(block) == BLOCK_DATA
    out = []
    for i in range(0, BLOCK_DATA, 5):
        b = block[i:i + 5]
        v = int.from_bytes(b, "little")  # 40 bits
        out += [v & 0xFFFFF, v >> 20]
    return out


def from_imm20(vals: list[int]) -> bytes:  # the guest's unpacking, for a self-test
    out = b""
    for i in range(0, 16, 2):
        v = vals[i] | vals[i + 1] << 20
        out += v.to_bytes(5, "little")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--size", default="80x60")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-seconds", type=float, default=None, help="encode only the first N seconds")
    ap.add_argument("--out", default=str(ROOT / "build"))
    ap.add_argument("--drom-max", type=int, default=DROM_MAX, help="bytes of DROM to use (test the sled)")
    a = ap.parse_args()
    w, h = map(int, a.size.split("x"))
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    f = frames(a.video, w, h, a.fps)
    if a.max_seconds:
        f = f[:int(a.max_seconds * a.fps)]
    p = pack(f)
    raw = p.tobytes()
    (out / "raw.bin").write_bytes(raw)
    print(f"{len(f)} frames of {w}x{h} @ {a.fps} fps: {len(raw)} bytes raw", flush=True)

    subprocess.run([str(ZX0), "-f", str(out / "raw.bin"), str(out / "video.zx0")], check=True,
                   capture_output=True)
    z = (out / "video.zx0").read_bytes()
    print(f"zx0: {len(z)} bytes ({len(z) / 1024:.1f} KB), ratio {len(raw) / len(z):.2f}x")

    drom = z[:a.drom_max]
    rest = z[a.drom_max:]
    rest += bytes(-len(rest) % BLOCK_DATA)
    n_blocks = len(rest) // BLOCK_DATA
    (out / "drom.bin").write_bytes(drom)
    lines = ["# the rest of the compressed video, as lui immediates (16 x 20 bits + ret per block)",
             "    .text", "    .globl sled_start", "    .p2align 2", "sled_start:"]
    for i in range(n_blocks):
        block = rest[i * BLOCK_DATA:(i + 1) * BLOCK_DATA]
        vals = to_imm20(block)
        assert from_imm20(vals) == block
        lines += [f"    lui {r}, {v:#x}" for r, v in zip(SLED_REGS, vals)]
        lines.append("    ret")
    (out / "sled.S").write_text("\n".join(lines) + "\n")
    (out / "config.h").write_text(
        f"#define FRAME_W {w}\n#define FRAME_H {h}\n#define FPS {a.fps}\n#define N_FRAMES {len(f)}\n"
        f"#define DROM_LEN {len(drom)}u\n#define SLED_BLOCKS {n_blocks}\n"
        f"#define SLED_BLOCK_BYTES {BLOCK_ROM}\n")
    print(f"DROM: {len(drom)} bytes; sled: {n_blocks} blocks = {len(rest)} data bytes in "
          f"{n_blocks * BLOCK_ROM} bytes of ROM ({n_blocks * BLOCK_ROM / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
