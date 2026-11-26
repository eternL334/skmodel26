#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ] || [ -z "${LSB_JOBID:-}" ]; then
    echo "Use submit.sh to run through LSF" >&2
    exit 1
fi
mode=$1
out=$2
cd "$(dirname "$0")"
root=$PWD
export OMP_DYNAMIC=FALSE
unset TASK1_SKIP_FIELD TASK1_PIN_CPU

{
    hostname
    printf 'job %s\nmode %s\nhosts %s\n' "$LSB_JOBID" "$mode" "${LSB_MCPU_HOSTS:-}"
    git rev-parse HEAD
    g++ --version
    lscpu -p=CPU,CORE,SOCKET,NODE
} > "$out/environment.txt"

cases=()
case "$mode" in
    seq)
        for m in 20 40 80; do cases+=("seq $m $m 1 1"); done
        cases+=("seq 400 600 1 1" "seq 800 1200 1 1")
        ;;
    omp)
        for t in 1 4 16; do cases+=("omp 80 80 1 $t"); done
        cases+=("seq 400 600 1 1" "seq 800 1200 1 1")
        for t in 1 2 4 8 16; do cases+=("omp 400 600 1 $t"); done
        for t in 1 4 8 16 32; do cases+=("omp 800 1200 1 $t"); done
        ;;
    mpi)
        for p in 1 2 4; do cases+=("mpi 80 80 $p 1"); done
        for p in 2 4 8 16; do cases+=("mpi 400 600 $p 1"); done
        for p in 4 8 16 32; do cases+=("mpi 800 1200 $p 1"); done
        ;;
    hybrid)
        for p in 1 2; do cases+=("hybrid 80 80 $p 4"); done
        for t in 1 2 4 8; do cases+=("hybrid 400 600 2 $t" "hybrid 800 1200 4 $t"); done
        ;;
    *) echo "Unknown mode: $mode" >&2; exit 1 ;;
esac

run_case() {
    local binary=$1 m=$2 n=$3 p=$4 t=$5 mpi=0
    local prefix="$out/run$trial/${m}x${n}_${binary}_p${p}_t${t}"
    local program="$root/build/poisson_$binary"
    local setting=(OMP_NUM_THREADS="$t" OMP_DYNAMIC=FALSE TASK1_PLACEMENT="$prefix")
    if (( trial > 1 )) || { (( m > 80 )) && [ "$mode" != seq ]; }; then
        setting+=(TASK1_SKIP_FIELD=1)
    fi
    if [ "$mode" = seq ] || { [ "$mode" = omp ] && (( t == 1 )); }; then
        setting+=(TASK1_PIN_CPU="${control_cpu:-first}")
    fi
    export OMP_NUM_THREADS=$t
    echo "Run $trial: $binary ${m}x${n}, P=$p T=$t"
    if [ "$mode" = seq ] || [ "$mode" = omp ]; then
        /polusfs/lsf/openmp/launchOpenMP.py /usr/bin/env "${setting[@]}" "$program" \
            "$m" "$n" 1e-8 "$prefix" > "${prefix}_summary.txt"
    else
        mpi=1
        mpiexec -n "$p" env "${setting[@]}" "$program" \
            "$m" "$n" 1e-8 "$prefix" > "${prefix}_summary.txt"
    fi
    if ! awk -v t="$t" -v p="$p" -v mpi="$mpi" '
        $1 == "threads" { threads = $2 }
        $1 == "threads_min" { lo = $2 }
        $1 == "threads_max" { hi = $2 }
        $1 == "processes" { ranks = $2 }
        $1 == "converged" { converged = $2 }
        END { exit !(converged == 1 && (mpi ? (ranks == p && lo == t && hi == t) : threads == t)) }
    ' "${prefix}_summary.txt"; then
        echo "Unexpected solver result in ${prefix}_summary.txt" >&2
        exit 1
    fi
    if [[ ( $mode = omp || $mode = seq ) && $t = 1 ]]; then
        local cpu
        cpu=$(awk 'NR == 2 && $4 == $5 {print $4}' "${prefix}_rank0_placement.txt")
        if [[ -z $cpu || $cpu != "${control_cpu:-$cpu}" ]]; then
            echo "One-thread control changed CPU in ${prefix}_rank0_placement.txt" >&2
            exit 1
        fi
        control_cpu=$cpu
    fi
}

count=${#cases[@]}
for trial in 1 2 3 4 5; do
    mkdir "$out/run$trial"
    for ((i=0; i<count; ++i)); do
        index=$(( (trial-1 + (trial%2 == 1 ? i : count-i)) % count ))
        read -r binary m n p t <<< "${cases[index]}"
        run_case "$binary" "$m" "$n" "$p" "$t"
    done
done

export LC_ALL=C
echo 'version M N processes threads min_seconds median_seconds max_seconds' > "$out/timings.txt"
for config in "${cases[@]}"; do
    read -r binary m n p t <<< "$config"
    awk '$1 == "seconds" {print $2}' "$out"/run[1-5]/"${m}x${n}_${binary}_p${p}_t${t}_summary.txt" |
        sort -g | awk -v config="$config" '
            {time[NR] = $1}
            END {if (NR != 5) exit 1; print config, time[1], time[3], time[5]}
        ' >> "$out/timings.txt"
done

if [ "$mode" = omp ]; then
    awk '
        BEGIN { print "M N threads speedup_vs_seq_same_node speedup_vs_omp1" }
        $2 < 400 || FNR == 1 {next}
        NR == FNR {
            if ($1 == "seq") seq[$2] = $7
            if ($1 == "omp" && $5 == 1) one[$2] = $7
            next
        }
        $1 == "omp" { print $2, $3, $5, seq[$2]/$7, one[$2]/$7 }
    ' "$out/timings.txt" "$out/timings.txt" > "$out/openmp_speedup.txt"
fi
