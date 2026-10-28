#ifndef MPI_SOLVER_H
#define MPI_SOLVER_H

#include "solver.h"
#include "partition.h"
#include <mpi.h>

struct LocalGrid {
    int M, N, rank, processes, px, py, cx, cy;
    int nx, ny, x0, y0, size, stride;
    int left, right, down, up;
    double hx, hy, eps;
    Range xr, yr;
    MPI_Comm comm;
    MPI_Datatype column;
    std::vector<double> theta, q, diag, b, halo;

    LocalGrid(int m, int n, MPI_Comm world);
    ~LocalGrid();
    LocalGrid(const LocalGrid &) = delete;
    LocalGrid &operator=(const LocalGrid &) = delete;
};

void exchange(LocalGrid &g);
void apply_A(LocalGrid &g, const std::vector<double> &v, std::vector<double> &out);
double dot(const LocalGrid &g, const std::vector<double> &a,
           const std::vector<double> &b);
Result solve(LocalGrid &g, double delta, int max_iter = 0);
std::vector<double> gather_solution(const LocalGrid &g, const std::vector<double> &v,
                                    std::vector<int> &layout);

#endif
