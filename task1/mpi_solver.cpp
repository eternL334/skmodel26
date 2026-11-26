#include "mpi_solver.h"
#include "pcg.h"
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
    inv_diag.resize(size);
    b.resize(size);
    halo.assign(static_cast<std::size_t>(slots),0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = j*nx+i;
            coefficients(x0+i,y0+j,M,N,hx,hy,eps,theta[k],q[k],diag[k],b[k]);
            inv_diag[k] = 1.0/diag[k];
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

namespace {
struct LocalStencil {
    LocalGrid &g;
    int nx, ny;
    double cx, cy;

    explicit LocalStencil(LocalGrid &grid) : g(grid), nx(g.nx), ny(g.ny),
        cx(1/(g.hx*g.hx)), cy(1/(g.hy*g.hy)) {}

    template<class F>
    void once(F f) const {
#ifdef _OPENMP
#pragma omp master
#endif
        { f(); }
#ifdef _OPENMP
#pragma omp barrier
#endif
    }

    void sum(double *values, int count) const {
        MPI_Allreduce(MPI_IN_PLACE,values,count,MPI_DOUBLE,MPI_SUM,g.comm);
    }

    void prepare(const std::vector<double> &v) const {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                g.halo[(j+1)*g.stride+i+1] = v[j*nx+i];
        once([&] { exchange(g); });
    }

    double value(const std::vector<double> &v, int i, int j) const {
        int k = j*nx+i;
        int c = (j+1)*g.stride+i+1;
        double s = g.diag[k]*v[k];
        s -= cx*g.halo[c-1];
        s -= cx*g.halo[c+1];
        s -= cy*g.halo[c-g.stride];
        s -= cy*g.halo[c+g.stride];
        return s;
    }
};
}

void apply_A(LocalGrid &g, const std::vector<double> &v, std::vector<double> &out) {
    apply_stencil(LocalStencil(g),v,out);
}

double dot(const LocalGrid &g, const std::vector<double> &a,
           const std::vector<double> &b) {
    double local = local_dot(a,b), total = 0;
    MPI_Allreduce(&local,&total,1,MPI_DOUBLE,MPI_SUM,g.comm);
    return total*g.hx*g.hy;
}

Result solve(LocalGrid &g, double delta, int max_iter) {
    return solve_pcg(g,LocalStencil(g),delta,max_iter);
}

std::vector<double> gather_solution(const LocalGrid &g, const std::vector<double> &v,
                                    std::vector<int> &layout, bool field) {
    int info[10] = {g.cx,g.cy,g.x0,g.y0,g.nx,g.ny,
                    g.xr.left,g.xr.right,g.yr.left,g.yr.right};
    layout.clear();
    if (g.rank == 0) layout.resize(10*g.processes);
    MPI_Gather(info,10,MPI_INT,g.rank == 0 ? layout.data() : 0,
               10,MPI_INT,0,g.comm);
    if (!field) return std::vector<double>();
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
