# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin 1.0 3
rundebug: run.c
	$(CC) -g -o run run.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations. 
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs. 
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific 
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens. 
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(CC) -Ofast -o run run.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run

.PHONY: win64
win64: 
	x86_64-w64-mingw32-gcc-win32 -Ofast -D_WIN32 -o run.exe -I. run.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run

.PHONY: clean
clean:
	rm -f run

# ----------------------------------------------------------------------------
# llama2-sycl (added by Muse Port): llama2.cpp, Muse's SYCL port of llama2.cu.
#   make llama2                        the CPU device (JIT, spir64)
#   make llama2 TARGET=nvidia          NVIDIA (intel/llvm with the CUDA backend)
#   make llama2 TARGET=amd AMD_ARCH=gfx950
#   ./llama2 stories15M.bin 0.0 64 "Once upon a time"
SYCL_CXX ?= icpx
TARGET ?= cpu
AMD_ARCH ?= gfx90a
ifeq ($(TARGET),nvidia)
  SYCL_TARGET_FLAGS = -fsycl-targets=nvptx64-nvidia-cuda
else ifeq ($(TARGET),amd)
  SYCL_TARGET_FLAGS = -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa \
                      --offload-arch=$(AMD_ARCH)
else
  SYCL_TARGET_FLAGS =
endif

llama2: llama2.cpp
	$(SYCL_CXX) -fsycl -O2 $(SYCL_TARGET_FLAGS) llama2.cpp -o llama2

.PHONY: clean-sycl
clean-sycl:
	rm -f llama2
