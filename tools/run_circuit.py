"""Free-run build/badapple.elf on the compiled circuit itself (the transformer layers that are
merged into the Qwenputer checkpoint) with Qwenputer's compact executor, checking every generated
token against the reference machine, which runs alongside.

usage: uv run --project qwenputer/experimental/riscv python tools/run_circuit.py
           [--start-frame F] [--frames N | --steps N] [--device cpu|cuda|mps] [--chunk 2000]

--start-frame F fast-forwards the reference machine to frame F and snapshots the circuit from its
state (registers, RAM, PC), so any part of the video can be checked on the circuit.
Writes build/circuit_<F>.bin (the bytes the circuit printed) and build/circuit_<F>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
from chipformer.rv import core, elf, fastexec, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

FB = 600  # bytes per frame (80x60 / 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--steps", type=int, default=None, help="stop after N tokens")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dense", action="store_true", help="dense matmuls (default: sparse rows)")
    ap.add_argument("--elf", default=str(ROOT / "build/badapple.elf"))
    a = ap.parse_args()

    prog = elf.program(a.elf)
    m = ref.Machine.from_program(prog)
    t0 = time.time()
    while not m.halted and len(m.out) < a.start_frame * FB:
        m.record()
    assert len(m.out) == a.start_frame * FB or m.halted
    print(f"reference fast-forwarded to frame {a.start_frame}: {m.steps} instructions, "
          f"pc {m.pc:#x}, {time.time() - t0:.0f}s", flush=True)
    byte_limit = None if a.frames is None else a.frames * FB

    t0 = time.time()
    c = core.build()
    fr = fastexec.FreeRun(c, 1, a.device, compile_=False, sparse=not a.dense)
    core.snapshot(c, m)
    fr.load(0, S.tok("GO"))
    print(f"circuit built and loaded in {time.time() - t0:.0f}s ({len(c.dims)} dims)", flush=True)

    want = []  # the reference's tokens after GO, generated as needed
    got = []
    out = bytearray()
    t0 = time.time()
    done = 0
    ok = True
    while True:
        if a.steps is not None and done >= a.steps:
            break
        if byte_limit is not None and len(out) >= byte_limit:
            break
        if m.halted and done >= len(want):
            break
        k = a.chunk if a.steps is None else min(a.chunk, a.steps - done)
        while len(want) < done + k and not m.halted:
            want += m.record()
        k = min(k, len(want) - done)
        if k <= 0:
            break
        toks = fr.run(k)[0]
        got.append(toks)
        bad = np.nonzero(toks != np.array(want[done:done + k], dtype=toks.dtype))[0]
        if len(bad):
            t = done + bad[0]
            print(f"MISMATCH at token {t + 1}: got {S.name(int(toks[bad[0]]))}, "
                  f"want {S.name(want[t])}")
            ok = False
            break
        for t in toks:
            fam, p = S.family(int(t))
            if fam == "BYTE" and p < 256:
                out.append(p)
        done += k
        el = time.time() - t0
        print(f"  {done} tokens, {len(out)} bytes ({len(out) / FB:.1f} frames), "
              f"{done / el:.0f} tok/s, {el:.0f}s", flush=True)
    tag = f"{a.start_frame}"
    (ROOT / f"build/circuit_{tag}.bin").write_bytes(out)
    raw = (ROOT / "build/raw.bin").read_bytes()
    start = a.start_frame * FB
    bytes_ok = bytes(out) == raw[start:start + len(out)]
    info = {"start_frame": a.start_frame, "tokens": done, "bytes": len(out),
            "frames": len(out) / FB, "tokens_match_reference": ok,
            "bytes_match_video": bytes_ok, "device": a.device,
            "seconds": time.time() - t0}
    (ROOT / f"build/circuit_{tag}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    sys.exit(0 if ok and bytes_ok else 1)


if __name__ == "__main__":
    main()
