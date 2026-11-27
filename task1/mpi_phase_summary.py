from __future__ import print_function
import os
import sys


PHASES = ('pack', 'borders', 'exchange_x', 'exchange_y', 'allreduce', 'stencil', 'vectors')


def read_rows(path):
    with open(path) as f:
        names = next(f).split()
        return [dict(zip(names, line.split())) for line in f if line.strip()]


def median(values):
    return sorted(values)[len(values)//2]


def analyze(folder, m, n, p, run, topology, ranks_file):
    base = os.path.join(folder, 'run%d' % run, '%dx%d_mpi-profile_p%d_t1' % (m,n,p))
    with open(base+'_summary.txt') as f:
        info = dict(line.split(None,1) for line in f if line.strip())
    it = int(info['iterations'])
    if (int(info['converged']) != 1 or int(info['restarts']) != 0 or
        it != {400:892, 800:1736}[m] or (int(info['M']), int(info['N']), int(info['processes'])) != (m,n,p)):
        raise ValueError('Unexpected problem or convergence in '+base)

    hosts, cores = set(), set()
    distinct_pinned_cores = True
    times, loops, solves = [], [], []
    cells = 0
    layout = read_rows(base+'_domains.dat')
    if len(layout) != p:
        raise ValueError('Unexpected process grid in '+base)
    for rank in range(p):
        rank_base = base+'_rank%d' % rank
        row, = read_rows(rank_base+'_phases.txt')
        placement, = read_rows(rank_base+'_placement.txt')
        cpu = int(placement['cpu'])
        host = placement['host']
        core, socket = topology[cpu]
        hosts.add(host)
        if (int(row['rank']) != rank or int(placement['rank']) != rank or
            int(placement['thread']) != 0 or int(row['iterations']) != it or
            cpu not in [int(c) for c in placement['allowed_cpus'].split(',')]):
            raise ValueError('Unexpected rank or thread in '+rank_base)
        allowed_cores = {topology[int(c)] for c in placement['allowed_cpus'].split(',')}
        if allowed_cores != {(core,socket)} or (socket,core) in cores:
            distinct_pinned_cores = False
        cores.add((socket,core))
        nx, ny = int(row['nx']), int(row['ny'])
        block = layout[rank]
        if (int(block['rank']), int(block['nx']), int(block['ny'])) != (rank,nx,ny):
            raise ValueError('Inconsistent rank layout in '+rank_base)
        cells += nx*ny
        solve, loop = float(row['solve_s']), float(row['loop_s'])
        phases = [float(row[key+'_s']) for key in PHASES]
        other = loop-sum(phases)
        if not (0 < loop <= solve < float('inf') and other >= 0 and
                all(0 <= t < float('inf') for t in phases)):
            raise ValueError('Invalid phase totals in '+rank_base)
        times.append(phases+[other])
        loops.append(loop)
        solves.append(solve)
        ranks_file.write(' '.join(map(str,[m,n,p,run,rank,host,cpu,core,socket,placement['allowed_cpus'],
                                           block['cx'],block['cy'],nx,ny,it,
                                           solve,loop]+phases+[other]))+'\n')

    if (len(hosts) != 1 or cells != (m-1)*(n-1) or
        abs(max(solves)-float(info['seconds'])) > 1e-8*max(solves)):
        raise ValueError('Inconsistent placement, block sizes or total time in '+base)
    if not distinct_pinned_cores:
        print('Warning: ranks not bound to distinct individual cores in '+base)
    phase_sums = [sum(col) for col in zip(*times)]
    avg_phase_ms = [1000*t/(p*it) for t in phase_sums]
    ns_per_node = [1e9*phase_sums[PHASES.index(k)]/(cells*it) for k in ('stencil','vectors')]
    return [1000*max(solves)/it,1000*max(loops)/it]+avg_phase_ms+ns_per_node


if __name__ == '__main__':
    folder, = sys.argv[1:]
    topology = {}
    with open(os.path.join(folder,'environment.txt'),'rb') as f:
        for line in f:
            cols = line.strip().split(b',')
            if len(cols) == 4 and cols[0].isdigit():
                topology[int(cols[0])] = (int(cols[1]),int(cols[2]))
    keys = list(PHASES)+['other']
    print('Medians of five runs. Solve and loop: slowest rank, ms/iteration. Phases: rank average, ms/iteration.')
    print('Last two columns: rank-summed compute time per interior grid point per iteration, ns.')
    print('M N P solve_ms loop_ms '+' '.join(k+'_ms' for k in keys)+' stencil_ns_point vectors_ns_point')
    with open(os.path.join(folder,'rank_phases.txt'),'w') as ranks:
        ranks.write('M N P run rank host cpu core socket allowed_cpus cx cy nx ny iterations solve_s loop_s '+
                    ' '.join(k+'_s' for k in keys)+'\n')
        for m,n,p in ((400,600,2),(400,600,4),(800,1200,8),(800,1200,16)):
            runs = [analyze(folder,m,n,p,i,topology,ranks) for i in range(1,6)]
            print(m,n,p,' '.join('%.5f' % median(col) for col in zip(*runs)))
