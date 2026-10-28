#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash submit.sh seq|omp|mpi|hybrid" >&2
    exit 1
fi
mode=$1
case "$mode" in
    seq) p=1; cores=1; span='span[hosts=1]' ;;
    omp) p=1; cores=16; span='span[hosts=1]' ;;
    mpi) p=32; cores=1; span='span[ptile=16]' ;;
    hybrid) p=4; cores=8; span='span[ptile=2]' ;;
    *) echo "Unknown mode: $mode" >&2; exit 1 ;;
esac

root=$(cd "$(dirname "$0")" && pwd)
cd "$root"
if ! command -v bsub >/dev/null; then
    echo "bsub not found. Run this script on Polus with LSF available." >&2
    exit 1
fi
if [ ! -x "build/poisson_$mode" ]; then
    echo "Run make $mode first" >&2
    exit 1
fi
if [ "$mode" = seq ] || [ "$mode" = omp ]; then
    if [ ! -x /polusfs/lsf/openmp/launchOpenMP.py ]; then
        echo "OpenMP launcher not found: /polusfs/lsf/openmp/launchOpenMP.py" >&2
        exit 1
    fi
else
    if ! command -v mpiexec >/dev/null; then
        echo "mpiexec not found. Run: module load SpectrumMPI" >&2
        exit 1
    fi
fi

out="$root/results/${mode}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$root/results"
mkdir "$out"
{
    printf '#!/bin/bash\n'
    printf '#BSUB -J task1_%s\n' "$mode"
    printf '#BSUB -q normal\n#BSUB -W 03:00\n'
    printf '#BSUB -n %s\n' "$p"
    printf '#BSUB -R "%s affinity[core(%s)]"\n' "$span" "$cores"
    printf 'exec /bin/bash'
    printf ' %q' "$root/run.sh" "$mode" "$out"
    printf '\n'
} > "$out/job.lsf"
echo "Results: $out"
bsub -oo "$out/job.out" -eo "$out/job.err" < "$out/job.lsf" | tee "$out/submitted.txt"
