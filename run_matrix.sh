#!/usr/bin/env bash
# Run on each GPU under test. Records GPU/driver/toolchain, the -O3 vs -O0
# verdict, and the per-arch SASS signal into results-<host>-<gpu>.txt.
set -u

NVCC=${NVCC:-nvcc}
CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
HOST=$(hostname)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' /' '__')
OUT="results-${HOST}-${GPU:-unknown}.txt"

{
  echo "==== ptxas uniform-counter miscompile: capture ===="
  echo "date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $HOST"
  echo
  echo "---- GPU / driver ----"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>&1
  echo
  echo "---- toolchain ----"
  $NVCC --version 2>&1 | tail -2
  ptxas --version 2>&1 | tail -2
  echo

  echo "---- build + run (native arch) ----"
  $NVCC -arch=native -std=c++17 -o repro_O3 repro.cu 2>&1
  $NVCC -arch=native -std=c++17 -Xptxas -O0 -o repro_O0 repro.cu 2>&1
  echo "[ptxas -O3 default]"
  ./repro_O3 2>&1
  O3=$?
  echo "exit: $O3"
  echo "[ptxas -O0]"
  ./repro_O0 2>&1
  O0=$?
  echo "exit: $O0"
  echo
  if [ "$O3" -eq 1 ] && [ "$O0" -eq 0 ]; then
    echo "VERDICT: REPRODUCED (O3 wrong, O0 correct)"
  elif [ "$O3" -eq 0 ] && [ "$O0" -eq 0 ]; then
    echo "VERDICT: not reproduced (both correct on this GPU/toolchain)"
  else
    echo "VERDICT: inconclusive (O3 exit=$O3, O0 exit=$O0)"
  fi
  echo

  echo "---- static per-arch SASS signal (counter promoted to uniform datapath?) ----"
  for A in sm_70 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120; do
    if $NVCC -arch=$A -std=c++17 -cubin -o /tmp/k_$A.cubin repro.cu 2>/dev/null; then
      SELUR=$($CUOBJDUMP -sass /tmp/k_$A.cubin 2>/dev/null | grep -cE "SEL R[0-9]+, R[0-9]+, UR[0-9]+")
      echo "  $A: SEL-from-uniform-counter = $SELUR (>=1 means promoted, bug likely)"
    else
      echo "  $A: not supported by this toolchain"
    fi
  done
} | tee "$OUT"

echo
echo "wrote $OUT"
