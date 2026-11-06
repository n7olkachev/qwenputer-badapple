"""Decode a .zx0 stream in Python (reference) and print literal/match statistics."""
import sys, collections
data = open(sys.argv[1], "rb").read()
pos = 0; last = 0; mask = 0; val = 0; back = False
def byte():
    global pos, last
    last = data[pos]; pos += 1; return last
def bit():
    global mask, val, back
    if back:
        back = False; return last & 1
    mask >>= 1
    if mask == 0:
        mask = 128; val = byte()
    return 1 if val & mask else 0
def gamma(inv):
    v = 1
    while not bit():
        v = v << 1 | (bit() ^ inv)
    return v
lits = []; matches = []; offs = collections.Counter(); out = 0; last_off = 1
state = "lit"
while True:
    if state == "lit":
        n = gamma(0); lits.append(n); out += n; pos += n
        state = "new" if bit() else "last"
    elif state == "last":
        n = gamma(0); matches.append(n); offs[last_off] += 1; out += n
        state = "lit" if not bit() else "new"
    else:
        last_off = gamma(1)
        if last_off == 256: break
        last_off = last_off * 128 - (byte() >> 1); back = True
        n = gamma(0) + 1; matches.append(n); offs[last_off] += 1; out += n
        state = "new" if bit() else "lit"
import statistics
print(f"output {out} bytes; {len(lits)} literal runs ({sum(lits)} bytes, mean {statistics.mean(lits):.1f}); "
      f"{len(matches)} matches ({sum(matches)} bytes, mean {statistics.mean(matches):.1f}, median {statistics.median(matches)})")
hist = collections.Counter(min(m, 64) // 8 * 8 for m in matches)
print("match length histogram (bucket: count, bytes):", {f"{k}+": (hist[k], sum(m for m in matches if min(m, 64) // 8 * 8 == k)) for k in sorted(hist)})
print("top offsets:", offs.most_common(8))
