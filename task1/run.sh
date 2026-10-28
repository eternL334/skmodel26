#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ] || [ -z "${LSB_JOBID:-}" ]; then
    echo "Use submit.sh to run through LSF" >&2
    exit 1
fi
mode=$1
out=$2
root=$(cd "$(dirname "$0")" && pwd)
cd "$root"
export OMP_DYNAMIC=FALSE
hostname
printf 'job %s\nmode %s\n' "$LSB_JOBID" "$mode"

run_case() {
    local n=$1 p=$2 t=$3 prefix
    prefix="$out/${n}x${n}_p${p}_t${t}"
    export OMP_NUM_THREADS=$t
    echo "Grid ${n}x${n}, processes $p, threads $t"
    if [ "$mode" = seq ] || [ "$mode" = omp ]; then
        /polusfs/lsf/openmp/launchOpenMP.py "$root/build/poisson_$mode" \
            "$n" "$n" 1e-8 "$prefix" > "${prefix}_summary.txt"
    else
        mpiexec -n "$p" "$root/build/poisson_$mode" \
            "$n" "$n" 1e-8 "$prefix" > "${prefix}_summary.txt"
    fi
    cat "${prefix}_summary.txt"
}

case "$mode" in
    seq)
        for n in 20 40 80; do run_case "$n" 1 1; done
        ;;
    omp)
        for t in 1 4 16; do run_case 80 1 "$t"; done
        ;;
    mpi)
        for p in 1 2 4; do run_case 80 "$p" 1; done
        ;;
    hybrid)
        for p in 1 2; do run_case 80 "$p" 4; done
        ;;
    *)
        echo "Unknown mode: $mode" >&2
        exit 1
        ;;
esac
