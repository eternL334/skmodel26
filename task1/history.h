#ifndef HISTORY_H
#define HISTORY_H

#include "solver.h"
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

inline void save_history(const Result &r, const std::string &prefix) {
    std::ofstream log((prefix + "_history.dat").c_str());
    if (!log) throw std::runtime_error("cannot open history file");
    log << std::setprecision(17);
    log << "iteration step residual energy restart\n";
    for (std::size_t k = 0; k < r.history.size(); ++k) {
        const Iteration &row = r.history[k];
        log << k+1 << ' ' << row.step << ' ' << row.residual << ' '
            << row.energy << ' ' << row.restart << '\n';
    }
    log.close();
    if (!log) throw std::runtime_error("failed to write history file");
}

#endif
