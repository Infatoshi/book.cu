NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing

# Target GPU architecture. These WGMMA kernels require Hopper (sm_90a, e.g. H100)
# and will not run on Ampere, Ada, or Blackwell. Override on the command line if
# needed, e.g. `make ARCH=sm_90`.
ARCH ?= sm_90a
NVCC_FLAGS += -arch=$(ARCH)

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

# Tensor core examples
tensor_cores: kernels/all_kernels.cu
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

profile: tensor_cores
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

clean:
	rm -rf $(OUT_DIR)/*

