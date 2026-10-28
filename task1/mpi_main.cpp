#include "mpi_solver.h"
#include <algorithm>
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

void save_mpi(const LocalGrid &g, const Result &r, const std::vector<double> &w,
              const std::vector<int> &layout, const std::string &prefix) {
    std::ofstream out((prefix+"_solution.dat").c_str());
    if (!out) throw std::runtime_error("cannot open solution file");
    out << std::setprecision(17) << "i j x y w inside theta\n";
    for (int j = 0; j <= g.N; ++j) {
        for (int i = 0; i <= g.M; ++i) {
            double x = -1+i*g.hx, y = -1+j*g.hy;
            double value = std::sin(x+y), theta = 0;
            if (i > 0 && i < g.M && j > 0 && j < g.N) {
                value = w[(j-1)*(g.M-1)+i-1];
                theta = cell_area(x,y,g.hx,g.hy)/(g.hx*g.hy);
                theta = std::max(0.0,std::min(1.0,theta));
            }
            bool inside = x >= 0 && y >= 0 && y <= 3 && x <= 3-y/3;
            out << i << ' ' << j << ' ' << x << ' ' << y << ' '
                << value << ' ' << inside << ' ' << theta << '\n';
        }
    }
    out.close();
    if (!out) throw std::runtime_error("failed to write solution file");

    std::ofstream log((prefix+"_history.dat").c_str());
    if (!log) throw std::runtime_error("cannot open history file");
    log << std::setprecision(17) << "iteration step residual energy restart\n";
    for (std::size_t k = 0; k < r.history.size(); ++k) {
        const Iteration &row = r.history[k];
        log << k+1 << ' ' << row.step << ' ' << row.residual << ' '
            << row.energy << ' ' << row.restart << '\n';
    }
    log.close();
    if (!log) throw std::runtime_error("failed to write history file");

    std::ofstream domains((prefix+"_domains.dat").c_str());
    if (!domains) throw std::runtime_error("cannot open domain layout file");
    domains << "rank cx cy i_first j_first nx ny left right bottom top geom_nx geom_ny\n";
    for (int rank = 0; rank < g.processes; ++rank) {
        domains << rank;
        for (int k = 0; k < 10; ++k) domains << ' ' << layout[10*rank+k];
        domains << ' ' << layout[10*rank+7]-layout[10*rank+6]+1
                << ' ' << layout[10*rank+9]-layout[10*rank+8]+1 << '\n';
    }
    domains.close();
    if (!domains) throw std::runtime_error("failed to write domain layout file");
}

int main(int argc, char **argv) {
    int required = MPI_THREAD_SINGLE, provided;
#ifdef _OPENMP
    required = MPI_THREAD_FUNNELED;
#endif
    MPI_Init_thread(&argc,&argv,required,&provided);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&processes);
    if (argc < 3 || argc > 5) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " M N [delta] [output_prefix]\n";
        MPI_Finalize();
        return 1;
    }
    int status = 0;
    try {
        if (provided < required)
            throw std::runtime_error("MPI thread support is insufficient");
        int threads = 1, openmp = 0, threads_min, threads_max;
#ifdef _OPENMP
        openmp = 1;
        omp_set_dynamic(0);
#pragma omp parallel
        {
#pragma omp single
            threads = omp_get_num_threads();
        }
#endif
        MPI_Allreduce(&threads,&threads_min,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        MPI_Allreduce(&threads,&threads_max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        int M = number<int>(argv[1]), N = number<int>(argv[2]);
        double delta = argc > 3 ? number<double>(argv[3]) : 1e-8;
        if (!std::isfinite(delta) || delta <= 0)
            throw std::invalid_argument("delta must be positive and finite");
        std::ostringstream name;
        name << "grid_" << M << '_' << N;
        if (openmp) name << "_hybrid" << processes << "_t" << threads;
        else name << "_mpi" << processes;
        std::string prefix = argc > 4 ? argv[4] : name.str();
        LocalGrid g(M,N,MPI_COMM_WORLD);
        MPI_Barrier(g.comm);
        double start = MPI_Wtime();
        Result r = solve(g,delta);
        double local_seconds = MPI_Wtime()-start, seconds = 0;
        MPI_Reduce(&local_seconds,&seconds,1,MPI_DOUBLE,MPI_MAX,0,g.comm);
        double local_area = 0, area = 0;
        for (int k = 0; k < g.size; ++k) local_area += g.theta[k]*g.hx*g.hy;
        MPI_Reduce(&local_area,&area,1,MPI_DOUBLE,MPI_SUM,0,g.comm);
        std::vector<int> layout;
        std::vector<double> full = gather_solution(g,r.w,layout);
        status = r.converged ? 0 : 2;
        if (rank == 0) {
            double step = r.history.empty() ? 0 : r.history.back().step;
            std::cout << std::setprecision(12)
                      << "M " << M << "\nN " << N
                      << "\nmpi 1\nprocesses " << processes
                      << "\nopenmp " << openmp
                      << "\nthreads_min " << threads_min << "\nthreads_max " << threads_max
                      << "\nmpi_thread_required " << required
                      << "\nmpi_thread_provided " << provided
                      << "\npx " << g.px << "\npy " << g.py
                      << "\nunknowns " << (M-1)*(N-1)
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
            save_mpi(g,r,full,layout,prefix);
        }
    } catch (const std::exception &e) {
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
