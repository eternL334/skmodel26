#include "solver.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

template <class T>
T number(const char *s) {
    std::istringstream in(s);
    T x;
    char tail;
    if (!(in >> x) || (in >> tail))
        throw std::invalid_argument("invalid numeric argument");
    return x;
}

void save(const Grid &g, const Result &r, const std::string &prefix) {
    std::ofstream out((prefix + "_solution.dat").c_str());
    if (!out) throw std::runtime_error("cannot open solution file");
    out << std::setprecision(17);
    out << "i j x y w inside theta\n";
    for (int j = 0; j <= g.N; ++j) {
        for (int i = 0; i <= g.M; ++i) {
            double x = g.x(i), y = g.y(j), t = 0;
            double w = std::sin(x+y);
            if (i > 0 && i < g.M && j > 0 && j < g.N) {
                w = r.w[g.id(i,j)];
                t = g.theta[g.id(i,j)];
            }
            bool inside = x >= 0 && y >= 0 && y <= 3 && x <= 3-y/3;
            out << i << ' ' << j << ' ' << x << ' ' << y << ' '
                << w << ' ' << inside << ' ' << t << '\n';
        }
    }
    out.close();
    if (!out) throw std::runtime_error("failed to write solution file");

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

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " M N [delta] [output_prefix]\n";
        return 1;
    }
    try {
        int M = number<int>(argv[1]), N = number<int>(argv[2]);
        double delta = argc > 3 ? number<double>(argv[3]) : 1e-8;
        int threads = 1, openmp = 0;
#ifdef _OPENMP
        openmp = 1;
        omp_set_dynamic(0);
#pragma omp parallel
        {
#pragma omp single
            threads = omp_get_num_threads();
        }
#endif
        std::ostringstream name;
        name << "grid_" << M << '_' << N;
        if (openmp) name << "_omp" << threads;
        std::string prefix = argc > 4 ? argv[4] : name.str();
        Grid g(M,N);
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        Result r = solve(g,delta);
        double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();

        double area = 0;
        for (int k = 0; k < g.size; ++k) area += g.theta[k]*g.hx*g.hy;
        double step = r.history.empty() ? 0 : r.history.back().step;
        std::cout << std::setprecision(12)
                  << "M " << M << "\nN " << N
                  << "\nopenmp " << openmp << "\nthreads " << threads
                  << "\nunknowns " << g.size
                  << "\nhx " << g.hx << "\nhy " << g.hy
                  << "\nepsilon " << g.eps << "\ndelta " << delta
                  << "\narea " << area
                  << "\niterations " << r.history.size()
                  << "\nrestarts " << r.restarts
                  << "\nstep_norm " << step
                  << "\ntrue_residual " << r.residual
                  << "\nrelative_residual " << r.relative_residual
                  << "\nseconds " << seconds
                  << "\nconverged " << r.converged << '\n';
        save(g,r,prefix);
        return r.converged ? 0 : 2;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
