/* Bad Apple!! on the Qwenputer: a ZX0 stream decoder running on the RV32IM CPU compiled into
 * Qwen3.5-4B's weights. The compressed video sits in the DROM (.rodata, 128 KB) and, past that,
 * in ROM as `lui` immediates (the "sled": blocks of 16 lui + ret, 40 data bytes per block, see
 * stub.S). Every decoded byte (8 horizontal pixels) is printed with putchar, so the trace of the
 * machine *is* the video.
 *
 * ZX0 decoder after Einar Saukas' dzx0_standard (BSD-3), made streaming. The match window is a
 * ring of RING *words* holding one byte each: on this machine a byte store (SB) costs 4 trace
 * tokens but a word store (SW) costs 1, and a word load of a small value is 1 token too. The
 * copy loop is unrolled by 8 (3 tokens per byte: lw, sw, ecall) and the bit reader has no
 * per-bit backtrack test (ZX0's one backtracked bit is handled where it happens). */
#include "config.h"

#define RING 6656u              /* bytes of window == the compressor's max offset (zx0_6656) */

extern const unsigned char drom_data[];                        /* .rodata -> DROM */
extern const unsigned char sled_start[];                       /* .text: the lui sled */
extern void sled_read(unsigned int *out16, const void *block); /* stub.S */

static unsigned int ring[RING];
static unsigned char sled_buf[40];

#define ALWAYS static inline __attribute__((always_inline))

/* putchar: ecall a7 = 1, a0 = the byte. No memory clobber: the machine only records a token. */
ALWAYS void put(unsigned int c) {
    register unsigned int a0 __asm__("a0") = c;
    register unsigned int a7 __asm__("a7") = 1;
    __asm__ volatile("ecall" : : "r"(a0), "r"(a7));
}

void exit(int code) {
    register int a0 __asm__("a0") = code;
    register int a7 __asm__("a7") = 93;
    for (;;) __asm__ volatile("ecall" : : "r"(a0), "r"(a7));
}

struct in {
    const unsigned char *p;        /* next DROM byte */
    unsigned int sled_n, sled_i;   /* bytes left / next in sled_buf */
    const unsigned char *sled_ptr;
    unsigned int mask, val;        /* the bit reader: current bit of val (128..1), 0 = empty */
};

static unsigned int refill_sled(struct in *in) __attribute__((noinline));
static unsigned int refill_sled(struct in *in) {
    unsigned int w[16];
    sled_read(w, in->sled_ptr);
    in->sled_ptr += SLED_BLOCK_BYTES;
    /* 16 x 20-bit immediates (held as imm << 12) -> 40 bytes: each pair is 5 bytes */
    unsigned char *p = sled_buf;
    for (int k = 0; k < 16; k += 2) {
        unsigned int v0 = w[k] >> 12, v1 = w[k + 1] >> 12;
        p[0] = v0 & 0xFF;
        p[1] = (v0 >> 8) & 0xFF;
        p[2] = ((v0 >> 16) & 0xF) | ((v1 & 0xF) << 4);
        p[3] = (v1 >> 4) & 0xFF;
        p[4] = (v1 >> 12) & 0xFF;
        p += 5;
    }
    in->sled_n = 39;
    in->sled_i = 1;
    return sled_buf[0];
}

ALWAYS unsigned int read_byte(struct in *in) {
    if (in->p != drom_data + DROM_LEN)
        return *in->p++;
    if (in->sled_n == 0)
        return refill_sled(in);
    in->sled_n--;
    return sled_buf[in->sled_i++];
}

ALWAYS unsigned int read_bit(struct in *in) {
    in->mask >>= 1;
    if (in->mask == 0) {
        in->mask = 128;
        in->val = read_byte(in);
    }
    return (in->val & in->mask) != 0;
}

/* interlaced Elias gamma; `value` starts at 1, or at the result of a first known bit */
ALWAYS unsigned int read_gamma(struct in *in, unsigned int inverted, unsigned int value) {
    while (!read_bit(in))
        value = value << 1 | (read_bit(in) ^ inverted);
    return value;
}

/* copy n >= 1 words src -> dst (no wrap inside), printing each byte */
ALWAYS void copy_chunk(unsigned int *dst, const unsigned int *src, unsigned int n) {
    register unsigned int a7 __asm__("a7") = 1;
    while (n >= 8) {
        __asm__ volatile(
            "lw a0, 0(%[s])\n\tsw a0, 0(%[d])\n\tecall\n\t"
            "lw a0, 4(%[s])\n\tsw a0, 4(%[d])\n\tecall\n\t"
            "lw a0, 8(%[s])\n\tsw a0, 8(%[d])\n\tecall\n\t"
            "lw a0, 12(%[s])\n\tsw a0, 12(%[d])\n\tecall\n\t"
            "lw a0, 16(%[s])\n\tsw a0, 16(%[d])\n\tecall\n\t"
            "lw a0, 20(%[s])\n\tsw a0, 20(%[d])\n\tecall\n\t"
            "lw a0, 24(%[s])\n\tsw a0, 24(%[d])\n\tecall\n\t"
            "lw a0, 28(%[s])\n\tsw a0, 28(%[d])\n\tecall"
            : : [s] "r"(src), [d] "r"(dst), "r"(a7) : "a0", "memory");
        src += 8;
        dst += 8;
        n -= 8;
    }
    while (n) {
        __asm__ volatile("lw a0, 0(%[s])\n\tsw a0, 0(%[d])\n\tecall"
                         : : [s] "r"(src), [d] "r"(dst), "r"(a7) : "a0", "memory");
        src++;
        dst++;
        n--;
    }
}

/* copy `len` bytes from `off` bytes back in the ring to the ring at dst, printing each; the
 * copies are chunked so the inner loop has no wrap test. Returns the new dst. */
ALWAYS unsigned int *copy_match(unsigned int *dst, unsigned int off, unsigned int len) {
    unsigned int *const end = ring + RING;
    /* (index arithmetic: the ring starts at 0x100, so dst - off would wrap below address 0) */
    unsigned int i = dst - ring;
    unsigned int *src = ring + (i >= off ? i - off : i + RING - off);
    for (;;) {
        unsigned int n = len;
        if (n > (unsigned int)(end - src)) n = end - src;
        if (n > (unsigned int)(end - dst)) n = end - dst;
        copy_chunk(dst, src, n);
        len -= n;
        src += n;
        dst += n;
        if (!len)
            return dst == end ? ring : dst;
        if (src == end) src = ring;
        if (dst == end) dst = ring;
    }
}

int main(void) {
    struct in in;                   /* (field by field: an aggregate initializer would put a
                                     constant in .rodata, which is all video) */
    in.p = drom_data;
    in.sled_n = in.sled_i = 0;
    in.sled_ptr = sled_start;
    in.mask = in.val = 0;
    unsigned int *dst = ring;
    unsigned int last_offset = 1, length, b;

copy_literals:
    length = read_gamma(&in, 0, 1);
    do {
        unsigned int c = read_byte(&in);
        *dst++ = c;
        put(c);
        if (dst == ring + RING) dst = ring;
    } while (--length);
    if (read_bit(&in))
        goto copy_from_new_offset;

    /* copy_from_last_offset: */
    length = read_gamma(&in, 0, 1);
    dst = copy_match(dst, last_offset, length);
    if (!read_bit(&in))
        goto copy_literals;

copy_from_new_offset:
    last_offset = read_gamma(&in, 1, 1);
    if (last_offset == 256)
        return 0;                       /* end of stream */
    b = read_byte(&in);
    last_offset = last_offset * 128 - (b >> 1);
    /* the length's gamma starts with bit 0 of that byte (ZX0's "backtrack") */
    length = 1;
    if (!(b & 1))
        length = read_gamma(&in, 0, 2 | read_bit(&in));
    length++;
    dst = copy_match(dst, last_offset, length);
    if (read_bit(&in))
        goto copy_from_new_offset;
    goto copy_literals;
}
