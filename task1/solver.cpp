#include "solver.h"
#include "pcg.h"
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
    inv_diag.resize(size);
    b.resize(size);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 1; j < N; ++j) {
        for (int i = 1; i < M; ++i) {
            int k = id(i,j);
            coefficients(i,j,M,N,hx,hy,eps,theta[k],q[k],diag[k],b[k]);
            inv_diag[k] = 1.0/diag[k];
        }
    }
}

namespace {
struct GridStencil : LocalReduction {
    const Grid &g;
    int nx, ny;
    double cx, cy;

    explicit GridStencil(const Grid &grid) : g(grid), nx(g.M-1), ny(g.N-1),
        cx(1/(g.hx*g.hx)), cy(1/(g.hy*g.hy)) {}

    void prepare(const std::vector<double> &) const {}

    double value(const std::vector<double> &v, int i, int j) const {
        int k = j*nx+i;
        double s = g.diag[k]*v[k];
        if (i > 0) s -= cx*v[k-1];
        if (i+1 < nx) s -= cx*v[k+1];
        if (j > 0) s -= cy*v[k-nx];
        if (j+1 < ny) s -= cy*v[k+nx];
        return s;
    }
};
}

void apply_A(const Grid &g, const std::vector<double> &v, std::vector<double> &out) {
    apply_stencil(GridStencil(g),v,out);
}

double local_dot(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0;
    int n = static_cast<int>(a.size());
#ifdef _OPENMP
#pragma omp parallel reduction(+:s)
#endif
    {
        KahanSum part;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int k = 0; k < n; ++k) part.add(a[k]*b[k]);
        s += part.value;
    }
    return s;
}

double dot(const Grid &g, const std::vector<double> &a,
           const std::vector<double> &b) {
    return local_dot(a,b)*g.hx*g.hy;
}

Result solve(const Grid &g, double delta, int max_iter) {
    return solve_pcg(g,GridStencil(g),delta,max_iter);
}
