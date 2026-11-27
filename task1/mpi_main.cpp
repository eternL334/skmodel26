#include "mpi_solver.h"
#include "mpi_profile.h"
#include "history.h"
#include "placement.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

template <class T>
T number(const char *s) {
    std::istringstream in(s);
    T x;
    char tail;
    if (!(in >> x) || (in >> tail))
        throw std::invalid_argument("invalid numeric argument");
    return x;
}

void save_solution(const LocalGrid &g, const std::vector<double> &w, const std::string &prefix) {
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
}

void save_domains(const LocalGrid &g, const std::vector<int> &layout, const std::string &prefix) {
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

#ifdef TASK1_MPI_PROFILE
void save_profile(const LocalGrid &g, const Result &r, double seconds,
                  const std::string &prefix) {
    const MpiProfile &t = mpi_profile();
    std::ostringstream path;
    path << prefix << "_rank" << g.rank << "_phases.txt";
    std::ofstream out(path.str().c_str());
    out << "rank nx ny iterations solve_s loop_s pack_s borders_s exchange_x_s exchange_y_s allreduce_s stencil_s vectors_s\n"
        << std::setprecision(17) << g.rank << ' ' << g.nx << ' ' << g.ny << ' '
        << r.history.size() << ' ' << seconds << ' ' << t.loop << ' ' << t.pack << ' '
        << t.borders << ' ' << t.exchange_x << ' ' << t.exchange_y << ' '
        << t.allreduce << ' ' << t.stencil << ' ' << t.vectors << '\n';
    out.close();
    if (!out) throw std::runtime_error("cannot write rank profile");
}
#endif

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
        int openmp = 0;
#ifdef _OPENMP
        openmp = 1;
#endif
        int threads = measure_team(rank), threads_min, threads_max;
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
#ifdef TASK1_MPI_PROFILE
        save_profile(g,r,local_seconds,prefix);
#endif
        double local_area = 0, area = 0;
        for (int k = 0; k < g.size; ++k) local_area += g.theta[k]*g.hx*g.hy;
        MPI_Reduce(&local_area,&area,1,MPI_DOUBLE,MPI_SUM,0,g.comm);
        std::vector<int> layout;
        bool field = !std::getenv("TASK1_SKIP_FIELD");
        std::vector<double> full = gather_solution(g,r.w,layout,field);
        status = r.converged ? 0 : 2;
        if (rank == 0) {
            double step = r.history.empty() ? 0 : r.history.back().step;
            std::cout << std::setprecision(12)
                      << "M " << M << "\nN " << N
                      << "\ncompiler " << __VERSION__
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
            save_history(r,prefix);
            save_domains(g,layout,prefix);
            if (field) save_solution(g,full,prefix);
        }
    } catch (const std::exception &e) {
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
