# the first DROM_LEN bytes of the compressed video: .rodata -> DROM (128 KB max)
    .section .rodata
    .globl drom_data
    .p2align 2
drom_data:
    .incbin "build/drom.bin"
