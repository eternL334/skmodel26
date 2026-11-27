import glob
import json
import os
import sys


def control_times(folder, version, reference):
    times = []
    pattern = os.path.join(folder, 'run[1-5]', '800x1200_%s_p1_t1_summary.txt' % version)
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            row = dict(line.split(None, 1) for line in f if line.strip())
        for key in ('M', 'N', 'unknowns', 'iterations', 'restarts'):
            if int(row[key]) != reference[key]:
                raise ValueError('Different %s in %s' % (key, path))
        if int(row['converged']) != 1 or not 0 <= float(row['step_norm']) < 1e-8:
            raise ValueError('Unfinished solve in %s' % path)
        times.append(float(row['seconds']))
    return times


def stats(times):
    if len(times) != 5 or not all(0 < t < float('inf') for t in times):
        raise ValueError('Expected five positive finite times')
    return sorted(times)[2], sum(times) / len(times)


if __name__ == '__main__':
    folder, measurements = sys.argv[1:]
    with open(measurements) as f:
        data = json.load(f)
    reference, = [row for row in data['configurations']
                  if (row['job'], row['version'], row['M'], row['N'], row['P'], row['T'])
                  == ('mpi', 'mpi', 800, 1200, 32, 1)]
    seq = stats(control_times(folder, 'seq', reference))
    mpi1 = stats(control_times(folder, 'mpi', reference))
    mpi32 = stats(reference['seconds'])
    print('MPI32 reference: Polus job %s' % data['jobs']['mpi']['id'])
    print('stat     seq_s     mpi1_s    mpi32_s   seq/mpi1   mpi1/mpi32')
    for i, label in enumerate(('median', 'mean')):
        print('%-6s %9.4f %9.4f %9.4f %10.3f %12.3f' %
              (label, seq[i], mpi1[i], mpi32[i], seq[i]/mpi1[i], mpi1[i]/mpi32[i]))
