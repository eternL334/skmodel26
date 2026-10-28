#include "solver.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

void near(double a, double b, double tol, const char *message) {
    check(std::isfinite(a) && std::abs(a-b) <= tol, message);
}

struct Point { double x, y; };

std::vector<Point> clip(const std::vector<Point> &p, double a, double b, double c) {
    std::vector<Point> out;
    for (std::size_t k = 0; k < p.size(); ++k) {
        Point u = p[k], v = p[(k+1)%p.size()];
        double fu = a*u.x+b*u.y-c, fv = a*v.x+b*v.y-c;
        if (fu <= 0) out.push_back(u);
        if ((fu <= 0) != (fv <= 0)) {
            double t = fu/(fu-fv);
            Point s = {u.x+t*(v.x-u.x), u.y+t*(v.y-u.y)};
            out.push_back(s);
        }
    }
    return out;
}

double polygon_area(double x, double y, double hx, double hy) {
    std::vector<Point> p = {{x-hx/2,y-hy/2}, {x+hx/2,y-hy/2},
                            {x+hx/2,y+hy/2}, {x-hx/2,y+hy/2}};
    p = clip(p,-1,0,0);
    p = clip(p,0,-1,0);
    p = clip(p,0,1,3);
    p = clip(p,3,1,9);
    double s = 0;
    for (std::size_t k = 0; k < p.size(); ++k) {
        Point a = p[k], b = p[(k+1)%p.size()];
        s += a.x*b.y-a.y*b.x;
    }
    return std::abs(s)/2;
}

std::vector<double> direct_solve(const Grid &g) {
    int n = g.size;
    std::vector<std::vector<double> > a(n,std::vector<double>(n,0));
    std::vector<double> b = g.b;
    for (int j = 1; j < g.N; ++j) {
        for (int i = 1; i < g.M; ++i) {
            int k = g.id(i,j);
            a[k][k] = g.diag[k];
            for (int jj = 1; jj < g.N; ++jj) {
                for (int ii = 1; ii < g.M; ++ii) {
                    int l = g.id(ii,jj);
                    if (j == jj && std::abs(i-ii) == 1) a[k][l] = -1/(g.hx*g.hx);
                    if (i == ii && std::abs(j-jj) == 1) a[k][l] = -1/(g.hy*g.hy);
                }
            }
        }
    }

    for (int k = 0; k < n; ++k) {
        int best = k;
        for (int i = k+1; i < n; ++i)
            if (std::abs(a[i][k]) > std::abs(a[best][k])) best = i;
        std::swap(a[k],a[best]);
        std::swap(b[k],b[best]);
        check(std::abs(a[k][k]) > 1e-15, "singular reference matrix");
        for (int i = k+1; i < n; ++i) {
            double t = a[i][k]/a[k][k];
            for (int j = k; j < n; ++j) a[i][j] -= t*a[k][j];
            b[i] -= t*b[k];
        }
    }
    std::vector<double> x(n);
    for (int i = n-1; i >= 0; --i) {
        double s = b[i];
        for (int j = i+1; j < n; ++j) s -= a[i][j]*x[j];
        x[i] = s/a[i][i];
    }
    return x;
}

int main() {
    try {
        std::vector<double> terms(200000,1e-6), ones(terms.size(),1);
        terms[0] = 1e8;
        near(local_dot(terms,ones),1e8+(terms.size()-1)*1e-6,2e-7,
             "small terms lost in a large dot product");
        near(cell_area(1,1,0.2,0.4),0.08,1e-14,"interior area");
        near(cell_area(-0.5,1,0.2,0.2),0,1e-14,"outside area");
        near(cell_area(0,0,0.2,0.2),0.01,1e-14,"corner area");
        near(cell_area(2.5,1.5,0.25,0.25),0.03125,1e-14,"sloping boundary area");
        near(cell_area(1.5,1.5,6,6),7.5,1e-14,"whole trapezoid area");
        for (int k = 0; k < 300; ++k) {
            double x = -0.5 + 4.0*((k*37)%300)/299;
            double y = -0.5 + 4.0*((k*83)%300)/299;
            double hx = 0.05 + 0.8*((k*17)%300)/299;
            double hy = 0.05 + 0.8*((k*53)%300)/299;
            near(cell_area(x,y,hx,hy),polygon_area(x,y,hx,hy),2e-14,"polygon comparison");
        }

        const int sizes[] = {20,40,80};
        for (int t = 0; t < 3; ++t) {
            Grid g(sizes[t],sizes[t]);
            double area = 0;
            for (int k = 0; k < g.size; ++k) {
                check(g.theta[k] >= 0 && g.theta[k] <= 1,"area fraction range");
                area += g.theta[k]*g.hx*g.hy;
            }
            near(area,7.5,2e-12,"total area");
        }

        Grid g(20,20);
        near(g.eps,0.0625,1e-15,"epsilon");
        near(g.q[g.id(1,1)],256,1e-12,"outside penalty");
        near(g.q[g.id(8,8)],0,1e-12,"inside penalty");
        near(g.q[g.id(14,10)],128,1e-12,"half-cell penalty");
        near(g.diag[g.id(14,10)],192,1e-12,"matrix diagonal");
        double b0 = 256*std::sin(-1.5) + 32*std::sin(-1.75);
        near(g.b[0],b0,1e-12,"Dirichlet contributions");

        Grid rect(12,17);
        std::vector<double> v(rect.size), av;
        double pi = std::acos(-1.0);
        double lambda = 2*(1-std::cos(pi/rect.M))/(rect.hx*rect.hx)
                      + 2*(1-std::cos(pi/rect.N))/(rect.hy*rect.hy);
        for (int j = 1; j < rect.N; ++j)
            for (int i = 1; i < rect.M; ++i)
                v[rect.id(i,j)] = std::sin(pi*i/rect.M)*std::sin(pi*j/rect.N);
        apply_A(rect,v,av);
        for (int k = 0; k < rect.size; ++k)
            near(av[k],(lambda+rect.q[k])*v[k],1e-12,"discrete sine operator");

        const int shapes[][2] = {{2,2},{7,9},{9,7}};
        for (int s = 0; s < 3; ++s) {
            Grid small(shapes[s][0],shapes[s][1]);
            Result r = solve(small,1e-12);
            std::vector<double> exact = direct_solve(small);
            check(r.converged,"PCG did not converge on a small grid");
            for (int k = 0; k < small.size; ++k)
                near(r.w[k],exact[k],2e-10,"PCG versus Gaussian elimination");
        }

        check(!solve(g,1e-20,1).converged,"iteration limit reported as success");
        std::fill(g.b.begin(),g.b.end(),0);
        Result zero = solve(g,1e-12);
        check(zero.converged && zero.history.empty(),"zero right-hand side");
        near(zero.residual,0,0,"zero residual");
        std::cout << "All tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
