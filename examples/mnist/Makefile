# Builds the MNIST digit-recognition example for libsparsr_hdc.
#
# It builds against the library in this repository and the Sparsr runtime from an unpacked
# SDK, named by SPARSR_SDK_ROOT, the same variable the library's own Makefile takes. The
# SDK is the free download from the Sparsr Developer Zone.
#
# There is no kernel to compile here. The device programs this uses are inside
# libsparsr_hdc.so already: the library compiles them when it is built and loads them into
# instruction memory when hdc_init() runs. So this needs no RISC-V toolchain: an ordinary
# host compiler is enough.

CC ?= gcc
CFLAGS ?= -O2 -g -Wall -Wextra -std=c11

SPARSR_SDK_ROOT ?=

HDC_DIR := ../..
HDC_INCLUDE := $(HDC_DIR)/include
HDC_BUILD_DIR := $(HDC_DIR)/build
HDC_LIB := $(HDC_BUILD_DIR)/libsparsr_hdc.so

BUILD_DIR := build
HOST_SRC := main.c
HOST_BIN := $(BUILD_DIR)/mnist_hdc

# Where to look for the MNIST files when `make run` is used. Override it on the command
# line: `make run MNIST_DIR=/path/to/mnist`.
MNIST_DIR ?= data

all: $(HOST_BIN)

.PHONY: check-sdk
check-sdk:
	@test -n "$(SPARSR_SDK_ROOT)" || { \
	  echo "SPARSR_SDK_ROOT is not set. This example links the Sparsr runtime from an unpacked"; \
	  echo "Sparsr SDK: download it from the Sparsr Developer Zone, unpack it, and run"; \
	  echo "  make SPARSR_SDK_ROOT=/path/to/sparsr-sdk"; \
	  exit 1; }

# Fail with something actionable rather than a linker error: the library is built from this
# repository, and this example cannot be built before it.
$(HDC_LIB):
	@echo "libsparsr_hdc.so has not been built yet. From the repository root:"
	@echo "  make SPARSR_SDK_ROOT=/path/to/sparsr-sdk all"
	@exit 1

# $ORIGIN reaches the library's own build directory relative to this binary; the SDK's
# lib/ is absolute, because a build tree and an SDK have no fixed relationship.
$(HOST_BIN): $(HOST_SRC) $(HDC_LIB) | check-sdk
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(HDC_INCLUDE) -I$(SPARSR_SDK_ROOT)/include -o $@ $(HOST_SRC) \
	    -L$(HDC_BUILD_DIR) -L$(SPARSR_SDK_ROOT)/lib \
	    -Wl,-rpath,'$$ORIGIN/../../../build:$(abspath $(SPARSR_SDK_ROOT))/lib' -Wl,--no-as-needed \
	    -lsparsr_hdc -lsparsr_host -lm -lstdc++ -pthread

# The library's kernels are RV32I, so they run on the Sparsr VM. The default softemu backend
# executes a different instruction set and would read those images as something else.
run: all
	SPARSR_BACKEND=vm SPARSR_MNIST_DIR=$(MNIST_DIR) ./$(HOST_BIN)

# What can be checked without the MNIST files, which are ~55 MB and in no repository: that
# the example builds, links against the real library, and reports missing data as missing
# data. Exit code 2 is the documented "no data" answer; a run that exited 0 here would look
# to a script exactly like a classifier that worked.
.PHONY: check
check: all
	@SPARSR_BACKEND=vm ./$(HOST_BIN) --data ./no-mnist-here >/dev/null 2>$(BUILD_DIR)/check.err; \
	  status=$$?; \
	  if [ "$$status" -ne 2 ]; then echo "Expected exit 2 with no data, got $$status."; cat $(BUILD_DIR)/check.err; exit 1; fi
	@grep -q "train-images-idx3-ubyte" $(BUILD_DIR)/check.err \
	  || { echo "The no-data message does not name the files it needs."; cat $(BUILD_DIR)/check.err; exit 1; }
	@echo "CHECK PASSED: the example builds and reports missing data correctly."

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run clean
