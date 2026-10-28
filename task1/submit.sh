#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash submit.sh seq|omp|mpi|hybrid" >&2
    exit 1
fi
mode=$1
case "$mode" in
    seq) p=1; t=1 ;;
    omp) p=1; t=16 ;;
    mpi) p=4; t=1 ;;
    hybrid) p=2; t=4 ;;
    *) echo "Unknown mode: $mode" >&2; exit 1 ;;
esac

root=$(cd "$(dirname "$0")" && pwd)
cd "$root"
command -v bsub >/dev/null
if [ ! -x "build/poisson_$mode" ]; then
    echo "Run make $mode first" >&2
    exit 1
fi
if [ "$mode" = seq ] || [ "$mode" = omp ]; then
    test -x /polusfs/lsf/openmp/launchOpenMP.py
else
    command -v mpiexec >/dev/null
fi

out="$root/results/${mode}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$root/results"
mkdir "$out"
{
    printf '#!/bin/bash\n'
    printf '#BSUB -J task1_%s\n' "$mode"
    printf '#BSUB -q short\n#BSUB -W 00:15\n'
    printf '#BSUB -n %s\n' "$p"
    printf '#BSUB -R "span[hosts=1] affinity[core(%s)]"\n' "$t"
    printf 'exec /bin/bash'
    printf ' %q' "$root/run.sh" "$mode" "$out"
    printf '\n'
} > "$out/job.lsf"
echo "Results: $out"
bsub -oo "$out/job.out" -eo "$out/job.err" < "$out/job.lsf" | tee "$out/submitted.txt"
