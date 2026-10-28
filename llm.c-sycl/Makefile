# llm.c-sycl: llm.c's fp32 GPT-2 trainer and test, in SYCL (ported by Muse; see README.md).
#   make                  build for the CPU device (JIT, spir64)
#   make TARGET=nvidia    ... for NVIDIA (intel/llvm with the CUDA backend)
#   make TARGET=amd AMD_ARCH=gfx950
# Needs ONEMATH_PREFIX: a oneMath install with the matching backend (MKL-CPU, cuBLAS, rocBLAS).
# Data: ./dev/download_starter_pack.sh (weights, tokenizer, tiny-shakespeare), as in llm.c.
SYCL_CXX ?= icpx
TARGET ?= cpu
AMD_ARCH ?= gfx90a
ifeq ($(TARGET),nvidia)
  TARGET_FLAGS = -fsycl-targets=nvptx64-nvidia-cuda
else ifeq ($(TARGET),amd)
  TARGET_FLAGS = -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa \
                 --offload-arch=$(AMD_ARCH)
else
  TARGET_FLAGS =
endif
ONEMATH_FLAGS = -I$(ONEMATH_PREFIX)/include -Wl,--no-as-needed -L$(ONEMATH_PREFIX)/lib \
                -Wl,-rpath,$(ONEMATH_PREFIX)/lib -lonemath
CXXFLAGS = -fsycl -O2 $(TARGET_FLAGS) -I.

TARGETS = train_gpt2fp32sycl
all: $(TARGETS)

train_gpt2fp32sycl: train_gpt2_fp32.cpp llmc/utils.h llmc/tokenizer.h llmc/dataloader.h llmc/rand.h
	$(SYCL_CXX) $(CXXFLAGS) train_gpt2_fp32.cpp -o $@ $(ONEMATH_FLAGS)

clean:
	rm -f $(TARGETS)
.PHONY: all clean
