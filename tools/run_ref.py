"""Run build/badapple.elf on Qwenputer's reference RV32 machine (the executable spec the circuit
is verified against), check the printed bytes are exactly build/raw.bin, and save the trace.

usage: uv run --project qwenputer/experimental/riscv python tools/run_ref.py [--frames N]
Writes build/trace.u32 (the token trace the circuit must reproduce) and build/out.bin.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
from chipformer.rv import elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402


def frame_bytes():
    cfg = dict(line.split()[1:3] for line in (ROOT / "build/config.h").read_text().splitlines())
    w, h = int(cfg["FRAME_W"]), int(cfg["FRAME_H"])
    return w * h // 8, w, h, int(cfg["FPS"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--elf", default=str(ROOT / "build/badapple.elf"))
    a = ap.parse_args()
    fb, w, h, fps = frame_bytes()
    raw = (ROOT / "build/raw.bin").read_bytes()
    limit = len(raw) if a.frames is None else min(len(raw), a.frames * fb)

    prog = elf.program(a.elf)
    m = ref.Machine.from_program(prog)
    toks = [S.tok("GO")]
    t0 = time.time()
    last = 0
    while not m.halted and len(m.out) < limit:
        toks += m.record()
        if len(m.out) - last >= fb * fps * 10:  # progress every 10 s of video
            last = len(m.out)
            print(f"  {len(m.out) // fb} frames, {m.steps} instr, {len(toks)} tokens, "
                  f"{time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    out = bytes(m.out)
    n_frames = len(out) // fb
    print(f"{n_frames} frames: {m.steps} instructions, {len(toks)} tokens "
          f"({len(toks) / max(m.steps, 1):.2f} tok/instr, {len(toks) / max(n_frames, 1):.0f} tok/frame), "
          f"{dt:.1f}s ({m.steps / dt:.0f} instr/s)")
    if m.fault:
        print("FAULT", S.FAULTS[m.fault[0]], hex(m.fault[1]))
    if m.halted:
        print("halted, exit code", m.exit_code)
    ok = out == raw[:len(out)] and len(out) >= limit
    print("output matches raw.bin:", ok, f"({len(out)} bytes)")
    (ROOT / "build/out.bin").write_bytes(out)
    np.array(toks, dtype=np.uint32).tofile(ROOT / "build/trace.u32")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
