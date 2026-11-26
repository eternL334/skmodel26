#ifndef PLACEMENT_H
#define PLACEMENT_H

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

inline int measure_team(int rank) {
    int threads = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);
    threads = omp_get_max_threads();
#endif
    const char *prefix = std::getenv("TASK1_PLACEMENT");
    bool record = prefix && *prefix;
    std::vector<std::string> rows(record ? threads : 0);
#ifdef __linux__
    char host[256] = {};
    if (record && gethostname(host,sizeof(host)-1) != 0)
        throw std::runtime_error("cannot read hostname");
    const char *pin = std::getenv("TASK1_PIN_CPU");
    int target = -1;
    if (pin && threads != 1) throw std::runtime_error("control run requires one thread");
    if (pin && std::string(pin) != "first") {
        char *end;
        long id = std::strtol(pin,&end,10);
        if (end == pin || *end || id < 0 || id >= CPU_SETSIZE)
            throw std::runtime_error("invalid control CPU");
        target = static_cast<int>(id);
    }
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
#pragma omp single
        threads = omp_get_num_threads();
#endif
#ifdef __linux__
        if (record) {
            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
            cpu_set_t mask;
            bool ok = sched_getaffinity(0,sizeof(mask),&mask) == 0;
            if (pin && ok) {
                int cpu = target < 0 ? sched_getcpu() : target;
                ok = cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu,&mask);
                if (ok) {
                    CPU_ZERO(&mask);
                    CPU_SET(cpu,&mask);
                    ok = sched_setaffinity(0,sizeof(mask),&mask) == 0
                      && sched_getaffinity(0,sizeof(mask),&mask) == 0;
                }
            }
            int cpu = sched_getcpu();
            if (ok && cpu >= 0) {
                std::ostringstream line;
                line << host << ' ' << rank << ' ' << thread << ' ' << cpu << ' ';
                bool first = true;
                for (int i = 0; i < CPU_SETSIZE; ++i) {
                    if (!CPU_ISSET(i,&mask)) continue;
                    if (!first) line << ',';
                    first = false;
                    line << i;
                }
                rows[thread] = line.str();
            }
        }
#endif
    }
    if (record) {
        std::ostringstream name;
        name << prefix << "_rank" << rank << "_placement.txt";
        std::ofstream file(name.str().c_str());
        file << "host rank thread cpu allowed_cpus\n";
        for (int t = 0; t < threads; ++t) {
            if (rows[t].empty()) throw std::runtime_error("cannot read or set Linux thread affinity");
            file << rows[t] << '\n';
        }
        file.close();
        if (!file) throw std::runtime_error("cannot write thread placement");
    }
    return threads;
}

#endif
