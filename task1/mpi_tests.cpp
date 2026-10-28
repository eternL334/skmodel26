#include "mpi_solver.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

void check_partition(int M, int N, int p) {
    Partition part = choose_partition(M,N,p);
    check(part.px*part.py == p,"wrong process grid size");
    std::vector<int> seen((M-1)*(N-1),0);
    int min_x = M, max_x = 0, min_y = N, max_y = 0;
    int min_gx = M+1, max_gx = 0, min_gy = N+1, max_gy = 0;
    for (int cy = 0; cy < part.py; ++cy) {
        Range y = split_axis(N,part.py,cy);
        if (cy == 0) check(y.left == 0,"missing bottom boundary");
        if (cy == part.py-1) check(y.right == N,"missing top boundary");
        if (cy > 0) check(y.left == split_axis(N,part.py,cy-1).right,"y domain gap");
        for (int cx = 0; cx < part.px; ++cx) {
            Range x = split_axis(M,part.px,cx);
            if (cx == 0) check(x.left == 0,"missing left boundary");
            if (cx == part.px-1) check(x.right == M,"missing right boundary");
            if (cx > 0) check(x.left == split_axis(M,part.px,cx-1).right,"x domain gap");
            int gx = x.right-x.left+1, gy = y.right-y.left+1;
            check(gx <= 2*gy && gy <= 2*gx,"domain aspect ratio");
            min_x = std::min(min_x,x.count); max_x = std::max(max_x,x.count);
            min_y = std::min(min_y,y.count); max_y = std::max(max_y,y.count);
            min_gx = std::min(min_gx,gx); max_gx = std::max(max_gx,gx);
            min_gy = std::min(min_gy,gy); max_gy = std::max(max_gy,gy);
            for (int j = y.first; j < y.first+y.count; ++j)
                for (int i = x.first; i < x.first+x.count; ++i)
                    ++seen[(j-1)*(M-1)+i-1];
        }
    }
    check(max_x-min_x <= 1 && max_y-min_y <= 1,"unbalanced owned nodes");
    check(max_gx-min_gx <= 1 && max_gy-min_gy <= 1,"unbalanced geometric domains");
    for (std::size_t k = 0; k < seen.size(); ++k)
        check(seen[k] == 1,"missing or duplicated unknown");
}

void check_grid(int M, int N) {
    LocalGrid g(M,N,MPI_COMM_WORLD);
    std::vector<double> v(g.size), av, one(g.size,1);
    double pi = std::acos(-1.0);
    double lambda = 2*(1-std::cos(pi/M))/(g.hx*g.hx)
                  + 2*(1-std::cos(pi/N))/(g.hy*g.hy);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            v[j*g.nx+i] = std::sin(pi*(g.x0+i)/M)*std::sin(pi*(g.y0+j)/N);
    std::fill(g.halo.begin(),g.halo.end(),123456);
    apply_A(g,v,av);
    for (int k = 0; k < g.size; ++k) {
        double expected = (lambda+g.q[k])*v[k];
        check(std::isfinite(av[k]) && std::abs(av[k]-expected) <= 1e-10*(1+std::abs(expected)),
              "MPI stencil or halo exchange");
    }
    double total = (M-1)*(N-1)*g.hx*g.hy;
    check(std::abs(dot(g,one,one)-total) < 1e-11,"global dot counts halo or loses nodes");
    std::vector<int> layout;
    std::vector<double> full_b = gather_solution(g,g.b,layout);
    std::vector<double> full_theta = gather_solution(g,g.theta,layout);
    Result r = solve(g,1e-8);
    check(r.converged,"MPI solver did not converge");
    int iterations = static_cast<int>(r.history.size()), lo, hi;
    MPI_Allreduce(&iterations,&lo,1,MPI_INT,MPI_MIN,g.comm);
    MPI_Allreduce(&iterations,&hi,1,MPI_INT,MPI_MAX,g.comm);
    check(lo == hi,"processes stopped at different iterations");
    std::vector<double> full = gather_solution(g,r.w,layout);
    if (g.rank == 0) {
#ifdef _OPENMP
        int threads = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        Grid reference(M,N);
        Result seq = solve(reference,1e-8);
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        check(seq.converged,"sequential reference did not converge");
        double error = 0;
        for (int k = 0; k < reference.size; ++k) {
            check(full_b[k] == reference.b[k],"MPI right-hand side differs");
            check(full_theta[k] == reference.theta[k],"MPI geometry differs");
            error = std::max(error,std::abs(full[k]-seq.w[k]));
        }
        check(error < 1e-6,"MPI solution differs from sequential");
        std::cout << M << 'x' << N << ", " << g.processes
                  << " processes, " << g.px << 'x' << g.py
                  << ", max difference " << error << '\n';
    }
    MPI_Barrier(g.comm);
    if (M >= 20 && N >= 20)
        check(!solve(g,1e-20,1).converged,"MPI iteration limit reported as success");
    std::fill(g.b.begin(),g.b.end(),0);
    Result zero = solve(g,1e-12);
    check(zero.converged && zero.history.empty() && zero.residual == 0,"MPI zero RHS");
}

int main(int argc, char **argv) {
    int required = MPI_THREAD_SINGLE, provided;
#ifdef _OPENMP
    required = MPI_THREAD_FUNNELED;
#endif
    MPI_Init_thread(&argc,&argv,required,&provided);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try {
        check(provided >= required,"insufficient MPI thread support");
#ifdef _OPENMP
        omp_set_dynamic(0);
        int threads = 0, expected = omp_get_max_threads();
#pragma omp parallel
        {
#pragma omp single
            threads = omp_get_num_threads();
        }
        check(threads == expected,"OpenMP team is smaller than requested");
        if (rank == 0) std::cout << "OpenMP threads per process: " << threads << '\n';
#endif
        const int p[] = {1,2,4,8,16,32};
        for (int i = 0; i < 6; ++i) {
            check_partition(80,80,p[i]);
            check_partition(400,600,p[i]);
            check_partition(800,1200,p[i]);
        }
        check_partition(81,77,6);
        check_partition(79,39,1);
        check_partition(3,3,4);
        check(choose_partition(80,80,2).px == 2,"unexpected square-grid partition");
        bool rejected = false;
        try { choose_partition(80,80,3); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected,"impossible aspect ratio accepted");
        rejected = false;
        try { choose_partition(2,2,2); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected,"empty subdomain accepted");
        check_grid(3,3);
        check_grid(12,17);
        check_grid(20,20);
        check_grid(81,77);
        if (rank == 0) std::cout << "All MPI tests passed\n";
    } catch (const std::exception &e) {
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
