#!/bin/bash
# Build guest/badapple.elf for the chipformer RV32 machine (ROM/DROM/RAM map, see env/rom.ld).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=qwenputer/experimental/riscv/chipformer/rv/env
GCC=riscv64-elf-gcc
$GCC -march=rv32im_zba -mabi=ilp32 -O2 -ffreestanding -fno-builtin -fno-asynchronous-unwind-tables -fno-unwind-tables -nostdlib -nostartfiles -static \
  -I"$ENV" -Ibuild -T"$ENV/rom.ld" -Wl,--no-warn-rwx-segments -o build/badapple.elf \
  "$ENV/crt0.S" guest/badapple.c guest/stub.S guest/drom.S build/sled.S
riscv64-elf-size build/badapple.elf
