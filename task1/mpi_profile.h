#ifndef MPI_PROFILE_H
#define MPI_PROFILE_H

#ifdef TASK1_MPI_PROFILE
#ifdef _OPENMP
#error The MPI phase profile requires one thread per rank
#endif
#include <mpi.h>

struct MpiProfile {
    double loop, pack, borders, exchange_x, exchange_y, allreduce, stencil, vectors;
    bool active;

    void start_loop() {
        *this = MpiProfile();
        active = true;
        loop = MPI_Wtime();
    }

    void end_loop() {
        loop = MPI_Wtime()-loop;
        active = false;
    }
};

inline MpiProfile &mpi_profile() {
    static MpiProfile profile = {};
    return profile;
}

#define PHASE_START(start) const double start = MPI_Wtime()
#define PHASE_END(start, field) do { if (mpi_profile().active) mpi_profile().field += MPI_Wtime()-(start); } while (0)
#else
#define PHASE_START(start) ((void)0)
#define PHASE_END(start, field) ((void)0)
#endif
#endif
