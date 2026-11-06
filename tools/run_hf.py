"""Check a slice of the video trace in the *transformer*: the chip layers emitted as a stock
Qwen3_5ForCausalLM (bf16, the same weights the merge puts in front of Qwen's layers), with the
machine state injected into the Gated DeltaNet recurrent states. Teacher-forced prefill; every
next-token prediction must equal the reference trace.

usage: uv run --project qwenputer/experimental/riscv python tools/run_hf.py [--start-frame F]
           [--tokens N] [--device cpu|cuda|mps] [--mode prefill|decode]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
from chipformer.rv import core, elf, hf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

FB = 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="prefill", choices=["prefill", "decode"])
    ap.add_argument("--elf", default=str(ROOT / "build/badapple.elf"))
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    m = ref.Machine.from_program(elf.program(a.elf))
    while not m.halted and len(m.out) < a.start_frame * FB:
        m.record()
    c = core.build()
    core.snapshot(c, m)
    toks = [S.tok("GO")]
    n0 = len(m.out)
    while len(toks) < a.tokens and not m.halted:
        toks += m.record()
    toks = toks[:a.tokens]
    out = bytes(m.out[n0:])
    print(f"frame {a.start_frame}: {len(toks)} tokens of trace, {len(out)} bytes printed in them",
          flush=True)

    t0 = time.time()
    torch.set_default_dtype(torch.bfloat16)  # build in bf16: ~4 GB, not 8
    model, lay, ids = hf.build_model(c, toks)
    torch.set_default_dtype(torch.float32)
    model = model.to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HF model: {n_params / 1e9:.2f}B params, hidden {lay.hidden}, "
          f"{model.config.num_hidden_layers} layers, built in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    positions, inexact, wrong = hf.check(c, model, lay, ids, toks, a.mode, tokens=True)
    el = time.time() - t0
    info = {"start_frame": a.start_frame, "tokens": positions, "inexact_dims": inexact,
            "wrong_predictions": wrong, "mode": a.mode, "device": a.device, "dtype": "bf16",
            "seconds": el}
    print(json.dumps(info))
    (ROOT / f"build/hf_{a.start_frame}.json").write_text(json.dumps(info, indent=1))
    sys.exit(0 if inexact == 0 and wrong == 0 else 1)


if __name__ == "__main__":
    main()
