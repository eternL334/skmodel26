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
    local m=$1 n=$2 p=$3 t=$4 prefix
    prefix="$out/${m}x${n}_p${p}_t${t}"
    export OMP_NUM_THREADS=$t
    echo "Grid ${m}x${n}, processes $p, threads $t"
    if [ "$mode" = seq ] || [ "$mode" = omp ]; then
        /polusfs/lsf/openmp/launchOpenMP.py "$root/build/poisson_$mode" \
            "$m" "$n" 1e-8 "$prefix" > "${prefix}_summary.txt"
    else
        mpiexec -n "$p" "$root/build/poisson_$mode" \
            "$m" "$n" 1e-8 "$prefix" > "${prefix}_summary.txt"
    fi
    cat "${prefix}_summary.txt"
}

case "$mode" in
    seq)
        run_case 400 600 1 1
        run_case 800 1200 1 1
        ;;
    omp)
        for t in 2 4 8 16; do run_case 400 600 1 "$t"; done
        for t in 4 8 16 32; do run_case 800 1200 1 "$t"; done
        ;;
    mpi)
        for p in 2 4 8 16; do run_case 400 600 "$p" 1; done
        for p in 4 8 16 32; do run_case 800 1200 "$p" 1; done
        ;;
    hybrid)
        for t in 1 2 4 8; do run_case 400 600 2 "$t"; done
        for t in 1 2 4 8; do run_case 800 1200 4 "$t"; done
        ;;
    *)
        echo "Unknown mode: $mode" >&2
        exit 1
        ;;
esac
