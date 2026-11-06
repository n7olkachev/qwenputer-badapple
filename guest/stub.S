# sled_read(unsigned int *out16, const void *block): call a sled block (16 x lui + ret, which
# leaves its 20-bit immediates in a0-a7, t0-t6, s1) and store the 16 registers to out16.
    .text
    .globl sled_read
sled_read:
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s1, 8(sp)
    sw   s0, 4(sp)
    mv   s0, a0           # out16
    jalr ra, a1, 0        # the block: lui x16, ret
    sw   a0,  0(s0)
    sw   a1,  4(s0)
    sw   a2,  8(s0)
    sw   a3, 12(s0)
    sw   a4, 16(s0)
    sw   a5, 20(s0)
    sw   a6, 24(s0)
    sw   a7, 28(s0)
    sw   t0, 32(s0)
    sw   t1, 36(s0)
    sw   t2, 40(s0)
    sw   t3, 44(s0)
    sw   t4, 48(s0)
    sw   t5, 52(s0)
    sw   t6, 56(s0)
    sw   s1, 60(s0)
    lw   s0, 4(sp)
    lw   s1, 8(sp)
    lw   ra, 12(sp)
    addi sp, sp, 16
    ret
