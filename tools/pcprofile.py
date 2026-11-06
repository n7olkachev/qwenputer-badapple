"""PC-level profile of build/badapple.elf on the reference machine: where the tokens go."""
import bisect, collections, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "qwenputer/experimental/riscv"))
from chipformer.rv import elf, ref, spec as S
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 600 * 60
m = ref.Machine.from_program(elf.program(ROOT / "build/badapple.elf"))
pcs, toks_by_pc, fam = collections.Counter(), collections.Counter(), collections.Counter()
while not m.halted and len(m.out) < limit:
    pc = m.pc; r = m.record(); pcs[pc] += 1; toks_by_pc[pc] += len(r)
    for t in r: fam[S.family(t)[0]] += 1
ntok = sum(fam.values())
print(f"instructions {m.steps}, tokens {ntok}, bytes {len(m.out)}: {ntok / len(m.out):.1f} tok/byte")
print("families:", fam.most_common(12))
dis = subprocess.run(["riscv64-elf-objdump", "-d", str(ROOT / "build/badapple.elf")], capture_output=True, text=True).stdout
dmap = {}
for l in dis.splitlines():
    p = l.split("\t")
    if len(p) >= 3 and p[0].strip().endswith(":"):
        dmap[int(p[0].strip()[:-1], 16)] = p[2].strip()
print("hottest instructions:")
for pc, c in pcs.most_common(40):
    print(f"  {pc:#07x} {c:8d} x {toks_by_pc[pc] / c:.1f} tok = {toks_by_pc[pc]:8d}  {dmap.get(pc, '?')}")
