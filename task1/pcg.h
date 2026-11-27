#ifndef PCG_H
#define PCG_H

#include "solver.h"
#include "mpi_profile.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

struct KahanSum {
    double value = 0, correction = 0;

    void add(double term) {
        double y = term - correction;
        double next = value + y;
        correction = (next - value) - y;
        value = next;
    }
};

struct LocalReduction {
    template<class F>
    void once(F f) const {
#ifdef _OPENMP
#pragma omp single
#endif
        { f(); }
    }

    void sum(double *, int) const {}
};

class TeamSums {
    std::vector<std::array<double,4> > parts;

public:
    TeamSums() : parts(1) {
#ifdef _OPENMP
        parts.resize(omp_get_max_threads());
#endif
    }

    template<int N, class Operator, class F>
    void finish(const Operator &op, const KahanSum (&local)[N], F done) {
        int thread = 0, threads = 1;
#ifdef _OPENMP
        thread = omp_get_thread_num();
        threads = omp_get_num_threads();
#endif
        for (int i = 0; i < N; ++i) parts[thread][i] = local[i].value;
#ifdef _OPENMP
#pragma omp barrier
#endif
        op.once([&] {
            double total[N] = {};
            for (int t = 0; t < threads; ++t)
                for (int i = 0; i < N; ++i) total[i] += parts[t][i];
            op.sum(total,N);
            done(total);
        });
    }
};

template<class Operator>
void apply_stencil(const Operator &op, const std::vector<double> &v,
                   std::vector<double> &out) {
    out.resize(v.size());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        op.prepare(v);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int j = 0; j < op.ny; ++j)
            for (int i = 0; i < op.nx; ++i)
                out[j*op.nx+i] = op.value(v,i,j);
    }
}

template<class GridType, class Operator>
Result solve_pcg(const GridType &g, const Operator &op, double delta, int max_iter) {
    if (!std::isfinite(delta) || delta <= 0)
        throw std::invalid_argument("delta must be positive and finite");
    int n = (g.M-1)*(g.N-1);
    if (max_iter == 0) max_iter = n;
    if (max_iter < 1 || max_iter > n)
        throw std::invalid_argument("invalid iteration limit");

    Result ans = {};
    ans.w.assign(g.size,0);
    std::vector<double> r = g.b;
    std::vector<double> z(g.size), p(g.size), ap(g.size), trial(g.size);
    TeamSums sums;
    const double cell = g.hx*g.hy;
    double rho = 0, alpha = 0, beta = 0, energy = 0, step = 0, bnorm = 0;
    bool restart = false;
    const char *error = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        KahanSum initial[2];
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
        for (int k = 0; k < g.size; ++k) {
            p[k] = r[k]*g.inv_diag[k];
            initial[0].add(r[k]*p[k]);
            initial[1].add(r[k]*r[k]);
        }
        sums.finish(op,initial,[&](const double *s) {
            rho = s[0]*cell;
            bnorm = std::sqrt(s[1]*cell);
            ans.converged = bnorm == 0;
        });

#ifdef TASK1_MPI_PROFILE
        mpi_profile().start_loop();
#endif
        for (int it = 0; it < max_iter && !ans.converged && !error; ++it) {
            op.prepare(p);
            PHASE_START(stencil_start);
            KahanSum direction[2];
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
            for (int j = 0; j < op.ny; ++j) {
                for (int i = 0; i < op.nx; ++i) {
                    int k = j*op.nx+i;
                    ap[k] = op.value(p,i,j);
                    direction[0].add(p[k]*ap[k]);
                    direction[1].add(p[k]*p[k]);
                }
            }
            PHASE_END(stencil_start,stencil);
            sums.finish(op,direction,[&](const double *s) {
                double den = s[0]*cell;
                if (!(den > 0) || !std::isfinite(den) || !(rho > 0) || !std::isfinite(rho)) {
                    error = "PCG breakdown";
                    return;
                }
                alpha = rho/den;
                step = std::abs(alpha)*std::sqrt(s[1]*cell);
            });
            if (error) break;

            for (int pass = 0; pass < 2; ++pass) {
                bool rollback = pass == 1;
                if (rollback) op.prepare(ans.w);
                PHASE_START(update_start);
                KahanSum state[4];
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
                for (int j = 0; j < op.ny; ++j) {
                    for (int i = 0; i < op.nx; ++i) {
                        int k = j*op.nx+i;
                        double w = ans.w[k];
                        if (rollback) {
                            r[k] = g.b[k] - op.value(ans.w,i,j);
                        } else {
                            w += alpha*p[k];
                            trial[k] = w;
                            r[k] -= alpha*ap[k];
                        }
                        z[k] = r[k]*g.inv_diag[k];
                        state[0].add(g.b[k]*w);
                        state[1].add(r[k]*w);
                        state[2].add(r[k]*r[k]);
                        state[3].add(r[k]*z[k]);
                    }
                }
                PHASE_END(update_start,vectors);
                sums.finish(op,state,[&](const double *s) {
                    double next = -0.5*(s[0]*cell + s[1]*cell);
                    if (!std::isfinite(next) || !std::isfinite(step) ||
                        !std::isfinite(s[2]) || !std::isfinite(s[3])) {
                        error = "non-finite PCG result";
                        return;
                    }
                    if (!rollback) {
                        double tol = 128*std::numeric_limits<double>::epsilon()
                                   * (1 + std::abs(energy) + std::abs(next));
                        restart = next > energy + tol;
                        if (restart) {
                            ++ans.restarts;
                            return;
                        }
                        ans.w.swap(trial);
                    }
                    energy = next;
                    double rr = s[2]*cell;
                    ans.history.push_back(Iteration{step,std::sqrt(rr),energy,restart});
                    ans.converged = rr == 0 || (!restart && step < delta);
                    double next_rho = s[3]*cell;
                    beta = restart ? 0 : next_rho/rho;
                    rho = next_rho;
                });
                if (error || !restart) break;
            }
            if (error || ans.converged || it+1 == max_iter) break;
            PHASE_START(direction_start);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int k = 0; k < g.size; ++k)
                p[k] = restart ? z[k] : z[k] + beta*p[k];
            PHASE_END(direction_start,vectors);
        }

#ifdef TASK1_MPI_PROFILE
        mpi_profile().end_loop();
#endif
        if (!error) {
            op.prepare(ans.w);
            KahanSum residual[1];
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
            for (int j = 0; j < op.ny; ++j) {
                for (int i = 0; i < op.nx; ++i) {
                    double d = g.b[j*op.nx+i] - op.value(ans.w,i,j);
                    residual[0].add(d*d);
                }
            }
            sums.finish(op,residual,[&](const double *s) {
                ans.residual = std::sqrt(s[0]*cell);
                ans.relative_residual = bnorm == 0 ? ans.residual : ans.residual/bnorm;
            });
        }
    }
    if (error) throw std::runtime_error(error);
    return ans;
}

#endif
