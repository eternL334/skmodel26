#include "mpi_solver.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

LocalGrid::LocalGrid(int m, int n, MPI_Comm world) : M(m), N(n) {
    MPI_Comm_size(world,&processes);
    Partition part = choose_partition(M,N,processes);
    px = part.px;
    py = part.py;
    int dims[2] = {py,px}, periods[2] = {0,0}, coords[2];
    MPI_Cart_create(world,2,dims,periods,0,&comm);
    MPI_Comm_rank(comm,&rank);
    MPI_Cart_coords(comm,rank,2,coords);
    cy = coords[0];
    cx = coords[1];
    MPI_Cart_shift(comm,1,1,&left,&right);
    MPI_Cart_shift(comm,0,1,&down,&up);
    xr = split_axis(M,px,cx);
    yr = split_axis(N,py,cy);
    x0 = xr.first;
    y0 = yr.first;
    nx = xr.count;
    ny = yr.count;
    long long slots = (static_cast<long long>(nx)+2)*(static_cast<long long>(ny)+2);
    if (slots > std::numeric_limits<int>::max())
        throw std::invalid_argument("local halo buffer is too large");
    size = nx*ny;
    stride = nx+2;
    hx = 5.0/M;
    hy = 5.0/N;
    double h = std::max(hx,hy);
    eps = h*h;
    theta.resize(size);
    q.resize(size);
    diag.resize(size);
    b.resize(size);
    halo.assign(static_cast<std::size_t>(slots),0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = j*nx+i;
            coefficients(x0+i,y0+j,M,N,hx,hy,eps,theta[k],q[k],diag[k],b[k]);
        }
    }
    MPI_Type_vector(ny,1,stride,MPI_DOUBLE,&column);
    MPI_Type_commit(&column);
}

LocalGrid::~LocalGrid() {
    if (std::uncaught_exception()) return;
    MPI_Type_free(&column);
    MPI_Comm_free(&comm);
}

void exchange(LocalGrid &g) {
    std::vector<double> &v = g.halo;
    int s = g.stride;
    for (int j = 1; j <= g.ny; ++j) {
        v[j*s] = 0;
        v[j*s+g.nx+1] = 0;
    }
    for (int i = 1; i <= g.nx; ++i) {
        v[i] = 0;
        v[(g.ny+1)*s+i] = 0;
    }
    MPI_Sendrecv(&v[s+1],1,g.column,g.left,10,
                 &v[s+g.nx+1],1,g.column,g.right,10,g.comm,MPI_STATUS_IGNORE);
    MPI_Sendrecv(&v[s+g.nx],1,g.column,g.right,11,
                 &v[s],1,g.column,g.left,11,g.comm,MPI_STATUS_IGNORE);
    MPI_Sendrecv(&v[s+1],g.nx,MPI_DOUBLE,g.down,12,
                 &v[(g.ny+1)*s+1],g.nx,MPI_DOUBLE,g.up,12,g.comm,MPI_STATUS_IGNORE);
    MPI_Sendrecv(&v[g.ny*s+1],g.nx,MPI_DOUBLE,g.up,13,
                 &v[1],g.nx,MPI_DOUBLE,g.down,13,g.comm,MPI_STATUS_IGNORE);
}

void apply_A(LocalGrid &g, const std::vector<double> &v, std::vector<double> &out) {
    out.resize(g.size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            g.halo[(j+1)*g.stride+i+1] = v[j*g.nx+i];
    exchange(g);
    double cx = 1/(g.hx*g.hx), cy = 1/(g.hy*g.hy);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            int k = j*g.nx+i;
            int c = (j+1)*g.stride+i+1;
            double s = g.diag[k]*v[k];
            s -= cx*g.halo[c-1];
            s -= cx*g.halo[c+1];
            s -= cy*g.halo[c-g.stride];
            s -= cy*g.halo[c+g.stride];
            out[k] = s;
        }
    }
}

double dot(const LocalGrid &g, const std::vector<double> &a,
           const std::vector<double> &b) {
    double local = 0, total = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:local) schedule(static)
#endif
    for (int k = 0; k < g.size; ++k) local += a[k]*b[k];
    MPI_Allreduce(&local,&total,1,MPI_DOUBLE,MPI_SUM,g.comm);
    return total*g.hx*g.hy;
}

Result solve(LocalGrid &g, double delta, int max_iter) {
    if (!std::isfinite(delta) || delta <= 0)
        throw std::invalid_argument("delta must be positive and finite");
    int total_size = (g.M-1)*(g.N-1);
    if (max_iter == 0) max_iter = total_size;
    if (max_iter < 1 || max_iter > total_size)
        throw std::invalid_argument("invalid iteration limit");
    Result ans;
    ans.w.assign(g.size,0);
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
        if (!(den > 0) || !std::isfinite(den) || !(rho > 0) || !std::isfinite(rho))
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
        double next_energy = -0.5*(dot(g,g.b,ans.w)+dot(g,r,ans.w));
        if (!std::isfinite(next_energy) || !std::isfinite(step))
            throw std::runtime_error("non-finite PCG result");
        double tol = 128*std::numeric_limits<double>::epsilon()
                   *(1+std::abs(energy)+std::abs(next_energy));
        bool restart = next_energy > energy+tol;
        if (restart) {
            ans.w.swap(old_w);
            apply_A(g,ans.w,ap);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < g.size; ++k) r[k] = g.b[k]-ap[k];
            next_energy = -0.5*(dot(g,g.b,ans.w)+dot(g,r,ans.w));
            ++ans.restarts;
        }
        energy = next_energy;
        double rr = dot(g,r,r);
        Iteration row = {step,std::sqrt(rr),energy,restart};
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
            for (int k = 0; k < g.size; ++k) p[k] = z[k]+beta*p[k];
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

std::vector<double> gather_solution(const LocalGrid &g, const std::vector<double> &v,
                                    std::vector<int> &layout) {
    int info[10] = {g.cx,g.cy,g.x0,g.y0,g.nx,g.ny,
                    g.xr.left,g.xr.right,g.yr.left,g.yr.right};
    layout.clear();
    if (g.rank == 0) layout.resize(10*g.processes);
    MPI_Gather(info,10,MPI_INT,g.rank == 0 ? layout.data() : 0,
               10,MPI_INT,0,g.comm);
    std::vector<int> counts, offsets;
    std::vector<double> packed, result;
    if (g.rank == 0) {
        counts.resize(g.processes);
        offsets.resize(g.processes);
        int offset = 0;
        for (int r = 0; r < g.processes; ++r) {
            counts[r] = layout[10*r+4]*layout[10*r+5];
            offsets[r] = offset;
            offset += counts[r];
        }
        packed.resize(offset);
        result.resize((g.M-1)*(g.N-1));
    }
    MPI_Gatherv(v.data(),g.size,MPI_DOUBLE,g.rank == 0 ? packed.data() : 0,
                g.rank == 0 ? counts.data() : 0,g.rank == 0 ? offsets.data() : 0,
                MPI_DOUBLE,0,g.comm);
    if (g.rank == 0) {
        for (int r = 0; r < g.processes; ++r) {
            int x0 = layout[10*r+2], y0 = layout[10*r+3];
            int nx = layout[10*r+4], ny = layout[10*r+5];
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    result[(y0+j-1)*(g.M-1)+x0+i-1] = packed[offsets[r]+j*nx+i];
        }
    }
    return result;
}
