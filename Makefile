# Builds libsparsr_hdc.so and its test harness.
#
# Two compilers run here, and keeping them straight is most of what this file does. The
# host library is ordinary C for this machine, linked against libsparsr_host. The kernels
# are RV32I for the Sparsr device, built by the stock RISC-V bare-metal GCC exactly as the
# SDK's own C-kernel example builds, then turned into raw images and embedded in the
# library as byte arrays.
#
# Embedding rather than shipping the images as data files is deliberate: an installed
# library would otherwise have to find them at run time, and the public host ABI can load
# instructions straight from memory.

CC ?= gcc
CFLAGS ?= -O2 -g -Wall -Wextra -Werror -std=c11 -fPIC
LDFLAGS ?=

RISCV_PREFIX ?= riscv64-unknown-elf-
RISCV_CC := $(RISCV_PREFIX)gcc
RISCV_OBJCOPY := $(RISCV_PREFIX)objcopy

# Where the SDK is: an unpacked Sparsr SDK tarball, named by SPARSR_SDK_ROOT. That is the
# one input this Makefile takes from outside the repository, and the tarball is the free
# download from the Sparsr Developer Zone. It holds the four things read here:
#
#   include/    sparsr.h, the host API, and sparsr_intrinsics.h, the kernel intrinsics
#   ldscripts/  sparsr.ld, the link script that places a kernel image
#   startup/    sparsr_crt0.S, the startup code a kernel image begins with
#   lib/        libsparsr_host.so and the backends it links
#
# With the variable unset the build stops and says so, in check-sdk below.
SPARSR_SDK_ROOT ?=
SDK_INCLUDE_DIRS := -I$(SPARSR_SDK_ROOT)/include
KERNEL_LDSCRIPT := $(SPARSR_SDK_ROOT)/ldscripts/sparsr.ld
KERNEL_STARTUP := $(SPARSR_SDK_ROOT)/startup/sparsr_crt0.S
SDK_LIB_DIR := $(SPARSR_SDK_ROOT)/lib
# The harness resolves the runtime where the SDK keeps it, beside itself first. Absolute,
# because a build tree and an SDK have no fixed relationship.
RPATH_TO_SPARSR_LIB := '$$ORIGIN:$(abspath $(SPARSR_SDK_ROOT))/lib'
# The library looks beside itself and nowhere else. Everywhere it is installed -- the SDK,
# the HDC tarball, the torchhd-sparsr wheel -- the runtime is in the same directory, and an
# absolute path baked in here is searched on every machine that loads the library and
# printed by readelf to anyone who asks. The published 0.2.0 wheel carried
# /repo/software/build/sdk-root/lib from the build container that way.
#
# So a program that loads this library from a build tree has to link libsparsr_host
# itself, with --no-as-needed, as the harness and the mnist example do below. The loader
# maps a program's own dependencies before it follows theirs, so the runtime is already
# loaded by the time this library asks for it. Without --no-as-needed, a toolchain that
# defaults to --as-needed (Debian's does) drops the runtime from the program's own list
# because the program calls nothing in it directly, and the library then looks beside
# itself and finds nothing.
LIB_RPATH := '$$ORIGIN'

BUILD_DIR := build
GEN_DIR := $(BUILD_DIR)/gen

LIB := $(BUILD_DIR)/libsparsr_hdc.so
TEST_BIN := $(BUILD_DIR)/test_hdc

# The device half. Same flags as the SDK's own C-kernel example, because a kernel this
# library builds is a kernel like any other: no C runtime, no startup files of the
# toolchain's own, and kernel_main placed at the start of the image.
KERNEL_CFLAGS := -march=rv32i -mabi=ilp32 -O2 -ffreestanding -nostdlib -nostartfiles \
                 -ffunction-sections -fno-pie -no-pie -Wall -Wextra

# Three kernels exchange scalars with the host at fixed data-memory addresses: the bundle
# kernel reads its operand count, the majority kernel reads a mode and writes a status, and
# the intersect kernel writes the overlap it reduced to. --defsym is how the SDK's own tests
# give a kernel a known place to read from.
#
# These are FLAT addresses: data memory is at 0x8000_0000 under the address map of decision
# 0017, and a kernel's pointer holds the whole address. They must match the
# HDC_DMEM_*_ADDRESS macros in src/device/hdc_device_layout.h, which is also where the
# region-relative word indices the host writes with are derived. Point one of these at the
# old 0x0 and the kernel names instruction memory, which the device traps.
#
# __bss_origin is a different kind of symbol: the link script reads it to decide where a
# kernel's own zero-initialised data goes. It defaults to the front of data memory, which is
# exactly where the five words above live, and the startup code clears .bss at the start of
# every batch — so a kernel that ever grew a `static` would silently wipe the operand count
# it had just been given. None of these kernels has one today. Setting the origin above the
# control block means none of them can, and it costs a flag. Keep it equal to
# HDC_DMEM_KERNEL_DATA_OFFSET in src/device/hdc_device_layout.h, and move both when a new
# control word is added.
KERNEL_DEFSYMS := -Wl,--defsym=hdc_operand_count=0x80000000 \
                  -Wl,--defsym=hdc_majority_mode=0x80000004 \
                  -Wl,--defsym=hdc_majority_total=0x80000008 \
                  -Wl,--defsym=hdc_majority_status=0x8000000C \
                  -Wl,--defsym=hdc_similarity_overlap=0x80000010 \
                  -Wl,--defsym=__bss_origin=0x80000014

KERNEL_NAMES := bind intersect bundle majority
KERNEL_IMAGES := $(patsubst %,$(GEN_DIR)/kernel_%.spex,$(KERNEL_NAMES))
KERNEL_HEADER := $(GEN_DIR)/hdc_kernel_images.h

all: $(LIB)

.PHONY: check-sdk
check-sdk:
	@test -n "$(SPARSR_SDK_ROOT)" || { \
	  echo "SPARSR_SDK_ROOT is not set. This library builds against an unpacked Sparsr SDK:"; \
	  echo "download it from the Sparsr Developer Zone, unpack it, and run"; \
	  echo "  make SPARSR_SDK_ROOT=/path/to/sparsr-sdk"; \
	  exit 1; }
	@test -f "$(SPARSR_SDK_ROOT)/include/sparsr_intrinsics.h" || { \
	  echo "SPARSR_SDK_ROOT=$(SPARSR_SDK_ROOT) does not look like an unpacked Sparsr SDK:"; \
	  echo "include/sparsr_intrinsics.h is missing."; \
	  exit 1; }

.PHONY: check-toolchain
check-toolchain:
	@command -v $(RISCV_CC) >/dev/null 2>&1 || { \
	  echo "$(RISCV_CC) not found. This library's kernels are RV32I and need the RISC-V bare-metal toolchain."; \
	  echo "On Debian or Ubuntu: sudo apt install gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf"; \
	  exit 1; }

$(GEN_DIR)/kernel_%.elf: src/device/kernel_%.c src/device/hdc_device_layout.h | check-sdk check-toolchain
	@mkdir -p $(GEN_DIR)
	$(RISCV_CC) $(KERNEL_CFLAGS) $(SDK_INCLUDE_DIRS) -Isrc/device \
	    -T$(KERNEL_LDSCRIPT) $(KERNEL_DEFSYMS) $(KERNEL_STARTUP) $< -o $@

$(GEN_DIR)/kernel_%.spex: $(GEN_DIR)/kernel_%.elf
	$(RISCV_OBJCOPY) -O binary -j .text $< $@

# One generated header holding every kernel image as a word array. The generator is a
# script rather than a shell one-liner so that what it emits stays readable and reviewable.
$(KERNEL_HEADER): $(KERNEL_IMAGES) scripts/embed_kernels.py
	@mkdir -p $(GEN_DIR)
	python3 scripts/embed_kernels.py --output $@ $(KERNEL_IMAGES)

$(BUILD_DIR)/sparsr_hdc.o: src/sparsr_hdc.c include/sparsr_hdc.h src/device/hdc_device_layout.h $(KERNEL_HEADER) | check-sdk
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -Isrc/device -I$(GEN_DIR) $(SDK_INCLUDE_DIRS) -c $< -o $@

$(LIB): $(BUILD_DIR)/sparsr_hdc.o
	$(CC) -shared -Wl,-soname,libsparsr_hdc.so -Wl,-rpath,$(LIB_RPATH) -o $@ $< \
	    -L$(SDK_LIB_DIR) -lsparsr_host -lm

# The harness sees src/device and the generated images as well as the public header: one
# case checks which instruction the majority kernel's ripple compiles to, which nothing the
# library returns can reveal. Every other case goes through include/ alone.
$(TEST_BIN): test/test_hdc.c $(LIB) $(KERNEL_HEADER)
	$(CC) $(CFLAGS) -Iinclude -Isrc/device -I$(GEN_DIR) $(SDK_INCLUDE_DIRS) -o $@ $< -L$(BUILD_DIR) -L$(SDK_LIB_DIR) \
	    -Wl,-rpath,$(RPATH_TO_SPARSR_LIB) -Wl,--no-as-needed \
	    -lsparsr_hdc -lsparsr_host -lm -lstdc++ -pthread

# The kernels are RV32I, so they run on the Sparsr VM. The default softemu backend
# executes a different instruction set and would read these images as something else.
.PHONY: test
test: $(TEST_BIN)
	SPARSR_BACKEND=vm ./$(TEST_BIN)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all
