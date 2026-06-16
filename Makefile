# test: run the kernel at ptxas -O3 (wrong) and -O0 (correct).
# probe: print GPU/driver/CUDA versions. sass: show the uniform-datapath diff.
# ARCH defaults to the build-time GPU; override e.g. ARCH=sm_90, ARCH=sm_120.

NVCC ?= nvcc
ARCH ?= native
NVCCFLAGS = -arch=$(ARCH) -lineinfo -std=c++17

all: repro_O3 repro_O0

repro_O3: repro.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

repro_O0: repro.cu
	$(NVCC) $(NVCCFLAGS) -Xptxas -O0 -o $@ $<

.PHONY: test probe clean sass
test: all
	@echo "===== ptxas -O3 (default) ====="; ./repro_O3; echo "exit: $$?"
	@echo "===== ptxas -O0 ====="; ./repro_O0; echo "exit: $$?"

# Dump SASS so the uniform-datapath difference is visible:
# -O3 promotes the candidate_position counter into a uniform register (UR),
# -O0 keeps it per-thread. Look for `SEL R*, R*, UR*, !P*` near the STG.
sass: all
	@echo "===== O3 uniform-datapath instructions ====="; cuobjdump -sass repro_O3 | grep -E "UIADD3 UR|USEL|UISETP|SEL R[0-9]+, R[0-9]+, UR" || true
	@echo "===== O0 uniform-datapath instructions (expect none) ====="; cuobjdump -sass repro_O0 | grep -E "UIADD3 UR|USEL|UISETP" || echo "(none)"

probe:
	@echo "nvidia-smi:"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
	@echo "nvcc:"; $(NVCC) --version | tail -2
	@echo "ptxas:"; ptxas --version | tail -2

clean:
	rm -f repro_O3 repro_O0 *.cubin *.ptx
