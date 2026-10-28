#include "solver.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

double cell_area(double x, double y, double hx, double hy) {
    double a = std::max(0.0, x - hx/2);
    double b = std::min(3.0, x + hx/2);
    double c = std::max(0.0, y - hy/2);
    double d = std::min(3.0, y + hy/2);
    if (a >= b || c >= d) return 0;

    double points[4] = {c, d, 0, 0};
    int n = 2;
    double t = 3*(3-b);
    if (t > c && t < d) points[n++] = t;
    t = 3*(3-a);
    if (t > c && t < d) points[n++] = t;
    std::sort(points, points+n);

    double s = 0;
    for (int k = 0; k < n-1; ++k) {
        double l = points[k], r = points[k+1];
        double wl = std::max(0.0, std::min(b, 3-l/3)-a);
        double wr = std::max(0.0, std::min(b, 3-r/3)-a);
        s += (wl+wr)*(r-l)/2;
    }
    return s;
}

int Grid::id(int i, int j) const {
    return (j-1)*(M-1) + i-1;
}

double Grid::x(int i) const { return -1 + i*hx; }
double Grid::y(int j) const { return -1 + j*hy; }

void coefficients(int i, int j, int M, int N, double hx, double hy, double eps,
                  double &theta, double &q, double &diag, double &b) {
    double xx = -1+i*hx, yy = -1+j*hy;
    double cx = 1/(hx*hx), cy = 1/(hy*hy);
    double t = cell_area(xx,yy,hx,hy)/(hx*hy);
    theta = std::max(0.0,std::min(1.0,t));
    q = (1-theta)/(eps*eps);
    diag = 2*cx + 2*cy + q;
    b = theta + q*std::sin(xx+yy);
    if (i == 1) b += cx*std::sin(-1+yy);
    if (i == M-1) b += cx*std::sin((-1+M*hx)+yy);
    if (j == 1) b += cy*std::sin(xx-1);
    if (j == N-1) b += cy*std::sin(xx+(-1+N*hy));
}

Grid::Grid(int m, int n) : M(m), N(n) {
    if (M < 2 || N < 2)
        throw std::invalid_argument("M and N must be at least 2");
    long long count = static_cast<long long>(M-1)*(N-1);
    if (count > std::numeric_limits<int>::max())
        throw std::invalid_argument("grid is too large");

    size = static_cast<int>(count);
    hx = 5.0/M;
    hy = 5.0/N;
    double h = std::max(hx, hy);
    eps = h*h;
    theta.resize(size);
    q.resize(size);
    diag.resize(size);
    b.resize(size);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 1; j < N; ++j) {
        for (int i = 1; i < M; ++i) {
            int k = id(i,j);
            coefficients(i,j,M,N,hx,hy,eps,theta[k],q[k],diag[k],b[k]);
        }
    }
}

void apply_A(const Grid &g, const std::vector<double> &v,
             std::vector<double> &out) {
    out.resize(g.size);
    double cx = 1/(g.hx*g.hx), cy = 1/(g.hy*g.hy);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 1; j < g.N; ++j) {
        for (int i = 1; i < g.M; ++i) {
            int k = g.id(i,j);
            double s = g.diag[k]*v[k];
            if (i > 1) s -= cx*v[k-1];
            if (i < g.M-1) s -= cx*v[k+1];
            if (j > 1) s -= cy*v[k-(g.M-1)];
            if (j < g.N-1) s -= cy*v[k+(g.M-1)];
            out[k] = s;
        }
    }
}

double dot(const Grid &g, const std::vector<double> &a,
           const std::vector<double> &b) {
    double s = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:s) schedule(static)
#endif
    for (int k = 0; k < g.size; ++k) s += a[k]*b[k];
    return s*g.hx*g.hy;
}

Result solve(const Grid &g, double delta, int max_iter) {
    if (!std::isfinite(delta) || delta <= 0)
        throw std::invalid_argument("delta must be positive and finite");
    if (max_iter == 0) max_iter = g.size;
    if (max_iter < 1 || max_iter > g.size)
        throw std::invalid_argument("invalid iteration limit");

    Result ans;
    ans.w.assign(g.size, 0);
    ans.converged = false;
    ans.restarts = 0;
    std::vector<double> r = g.b;
    std::vector<double> z(g.size), p(g.size), ap(g.size), old_w(g.size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < g.size; ++k) {
        z[k] = r[k]/g.diag[k];
        p[k] = z[k];
    }
    double rho = dot(g,r,z);
    double energy = 0;
    if (dot(g,r,r) == 0) ans.converged = true;

    for (int it = 0; it < max_iter && !ans.converged; ++it) {
        apply_A(g,p,ap);
        double den = dot(g,p,ap);
        if (!(den > 0) || !std::isfinite(den) ||
            !(rho > 0) || !std::isfinite(rho))
            throw std::runtime_error("PCG breakdown");

        double alpha = rho/den;
        double step = std::abs(alpha)*std::sqrt(dot(g,p,p));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int k = 0; k < g.size; ++k) {
            old_w[k] = ans.w[k];
            ans.w[k] += alpha*p[k];
            r[k] -= alpha*ap[k];
        }

        double next_energy = -0.5*(dot(g,g.b,ans.w) + dot(g,r,ans.w));
        if (!std::isfinite(next_energy) || !std::isfinite(step))
            throw std::runtime_error("non-finite PCG result");
        double tol = 128*std::numeric_limits<double>::epsilon()
                   * (1 + std::abs(energy) + std::abs(next_energy));
        bool restart = next_energy > energy + tol;

        if (restart) {
            ans.w.swap(old_w);
            apply_A(g,ans.w,ap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < g.size; ++k) r[k] = g.b[k]-ap[k];
            next_energy = -0.5*(dot(g,g.b,ans.w) + dot(g,r,ans.w));
            ++ans.restarts;
        }

        energy = next_energy;
        double rr = dot(g,r,r);
        Iteration row = {step, std::sqrt(rr), energy, restart};
        ans.history.push_back(row);
        if (rr == 0 || (!restart && step < delta)) {
            ans.converged = true;
            break;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int k = 0; k < g.size; ++k) z[k] = r[k]/g.diag[k];
        double next_rho = dot(g,r,z);
        if (restart) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < g.size; ++k) p[k] = z[k];
        } else {
            double beta = next_rho/rho;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < g.size; ++k) p[k] = z[k] + beta*p[k];
        }
        rho = next_rho;
    }

    apply_A(g,ans.w,ap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < g.size; ++k) r[k] = g.b[k]-ap[k];
    ans.residual = std::sqrt(dot(g,r,r));
    double bn = std::sqrt(dot(g,g.b,g.b));
    ans.relative_residual = bn == 0 ? ans.residual : ans.residual/bn;
    return ans;
}
