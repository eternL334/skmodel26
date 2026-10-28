#ifndef SOLVER_H
#define SOLVER_H

#include <vector>

struct Grid {
    int M, N, size;
    double hx, hy, eps;
    std::vector<double> theta, q, diag, b;

    Grid(int m, int n);
    int id(int i, int j) const;
    double x(int i) const;
    double y(int j) const;
};

struct Iteration {
    double step, residual, energy;
    bool restart;
};

struct Result {
    std::vector<double> w;
    std::vector<Iteration> history;
    bool converged;
    int restarts;
    double residual, relative_residual;
};

double cell_area(double x, double y, double hx, double hy);
void coefficients(int i, int j, int M, int N, double hx, double hy, double eps,
                  double &theta, double &q, double &diag, double &b);
double local_dot(const std::vector<double> &a, const std::vector<double> &b);
double dot(const Grid &g, const std::vector<double> &a,
           const std::vector<double> &b);
void apply_A(const Grid &g, const std::vector<double> &v,
             std::vector<double> &out);
Result solve(const Grid &g, double delta, int max_iter = 0);

#endif
