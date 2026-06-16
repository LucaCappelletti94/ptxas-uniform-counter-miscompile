# ptxas uniform-datapath miscompilation

`ptxas -O>=1` miscompiles a kernel whose outer loop has a warp-uniform trip count but a per-thread divergent body. It hoists the loop counter into a uniform register (`UR`), and a per-thread select then reads that counter incoherently, so one thread records an out-of-range index. Correct at `ptxas -O0`, wrong at `-O1`/`-O2`/`-O3`. It reproduces from PTX emitted by both `nvcc` and `nvrtc`. One `.cu`, a 160 KB fixture, no framework dependencies.

## Run

```bash
make test     # builds at -O3 and -O0, runs both
```

Each thread loops `candidate_position` over `0..candidate_count` (4, a warp-uniform scalar argument), scores candidate partner rows, and stores the best index, so the result is always in `[0, 4)`:

```c
for (unsigned candidate_position = 0; candidate_position < candidate_count;
     ++candidate_position) {
    float score = modified_linear_cosine_score_rows(/* ... */, peaks);
    if (score > best_score) { best_score = score; best_position = candidate_position; }
}
best_position_out[anchor] = (int)best_position;
```

At `-O3`, anchor 13673 gets `best_position = 4`, the trip count, impossible from the loop body (one thread out of 32768). At `-O0` it gets `2`. `make probe` prints versions, `make sass` shows the SASS diff.

## Root cause

`ptxas -O3` promotes the counter into the uniform datapath. The trip count is warp-uniform so this looks legal, but the body diverges per thread (the scorer has data-dependent inner loops and branches), so a lagging thread reads the post-loop counter (`4`) through a per-thread-predicated select.

```
# -O0 (correct): per-thread counter R7
SEL   R28, R7,  R28, P0      # best_position = P0 ? candidate_position(R7) : best_position
IADD3 R7,  R7,  0x1, RZ      # counter++ (per-thread)
ISETP.LT.U32 P0, R7, R6      # per-thread guard

# -O3 (wrong): counter promoted to uniform UR4
SEL    R10, R10, UR4, !P0    # best_position reads UNIFORM counter UR4
UIADD3 UR4, UR4, 0x1, URZ    # counter++ (uniform)
UISETP.GE.U32 UP0, UR4, UR5  # uniform guard
```

The defect is in `ptxas`, not the front end or the kernel:

1. The same `nvrtc` PTX, JIT-loaded with `cuModuleLoadDataEx`, is correct at `CU_JIT_OPTIMIZATION_LEVEL 0` and wrong at 1 to 4. The PTX is byte-identical, so only the `ptxas` level varies.
2. Standalone `ptxas -O0` gives a correct cubin, `ptxas -O3` a wrong one, from that PTX.
3. `compute-sanitizer --tool memcheck` is clean on the wrong build, so this is codegen, not an out-of-bounds access.

## Trigger sensitivity

- `peaks` (tensor peak count) is a runtime argument. As a compile-time constant the inner loops simplify, the counter is not promoted, and the bug vanishes.
- Replacing the inner conflict-graph dynamic program with a plain sum also stops the promotion.
- The corruption depends on the offender's whole 32-thread warp, so the fixture ships exactly the 159 rows that warp reads (the rest are generated as zeros). `batch_items` must stay 32768 because the partner schedule derives from it.

## Coverage

Reproduced on two RTX 4090 hosts (sm_89): driver 595.71.05 with CUDA 13.2, and driver 535.309.01 with CUDA 12.0. The uniform-counter `SEL R*, R*, UR*` also appears in the cubin for sm_80, sm_86, sm_90, sm_100, and sm_120, so this is not limited to Ada. Run `bash run_matrix.sh` to confirm on a given GPU.

## Files

- `repro.cu`: kernel (a hand transcription of a CubeCL-generated kernel with a plain pointer ABI and literal shapes) plus the host harness, which builds zeroed `[32768, 128]` arrays and injects the 159 real rows.
- `spectra.bin`: the fixture (header `u32 num_rows, num_peaks, batch_items`, then 159 records of `u32 row_index`, `f32 mz[128]`, `f32 intensity[128]`, `f32 precursor`, little-endian).
- `Makefile`, `run_matrix.sh`: build/run and cross-GPU capture.
