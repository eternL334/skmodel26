CXX = g++
MPICXX = mpicxx
CXXFLAGS = -O2 -std=c++11 -Wall -Wextra -pedantic
MPIEXEC = mpiexec

.PHONY: all seq omp mpi hybrid test test-omp test-mpi test-hybrid clean

all: seq omp mpi hybrid

seq: build/poisson_seq
omp: build/poisson_omp
mpi: build/poisson_mpi
hybrid: build/poisson_hybrid

build:
	mkdir -p build

build/poisson_seq: main.cpp solver.cpp solver.h | build
	$(CXX) $(CXXFLAGS) main.cpp solver.cpp -o $@

build/poisson_omp: main.cpp solver.cpp solver.h | build
	$(CXX) $(CXXFLAGS) -fopenmp main.cpp solver.cpp -o $@

build/poisson_mpi: mpi_main.cpp mpi_solver.cpp mpi_solver.h partition.cpp partition.h solver.cpp solver.h | build
	$(MPICXX) $(CXXFLAGS) mpi_main.cpp mpi_solver.cpp partition.cpp solver.cpp -o $@

build/poisson_hybrid: mpi_main.cpp mpi_solver.cpp mpi_solver.h partition.cpp partition.h solver.cpp solver.h | build
	$(MPICXX) $(CXXFLAGS) -fopenmp mpi_main.cpp mpi_solver.cpp partition.cpp solver.cpp -o $@

build/tests: tests.cpp solver.cpp solver.h | build
	$(CXX) $(CXXFLAGS) tests.cpp solver.cpp -o $@

build/tests_omp: tests.cpp solver.cpp solver.h | build
	$(CXX) $(CXXFLAGS) -fopenmp tests.cpp solver.cpp -o $@

build/tests_mpi: mpi_tests.cpp mpi_solver.cpp mpi_solver.h partition.cpp partition.h solver.cpp solver.h | build
	$(MPICXX) $(CXXFLAGS) mpi_tests.cpp mpi_solver.cpp partition.cpp solver.cpp -o $@

build/tests_hybrid: mpi_tests.cpp mpi_solver.cpp mpi_solver.h partition.cpp partition.h solver.cpp solver.h | build
	$(MPICXX) $(CXXFLAGS) -fopenmp mpi_tests.cpp mpi_solver.cpp partition.cpp solver.cpp -o $@

test: build/tests
	./build/tests

test-omp: build/tests_omp
	OMP_NUM_THREADS=1 ./build/tests_omp
	OMP_NUM_THREADS=4 ./build/tests_omp
	OMP_NUM_THREADS=16 ./build/tests_omp

test-mpi: build/tests_mpi
	$(MPIEXEC) -n 1 ./build/tests_mpi
	$(MPIEXEC) -n 2 ./build/tests_mpi
	$(MPIEXEC) -n 4 ./build/tests_mpi

test-hybrid: build/tests_hybrid
	OMP_NUM_THREADS=4 $(MPIEXEC) -n 1 ./build/tests_hybrid
	OMP_NUM_THREADS=4 $(MPIEXEC) -n 2 ./build/tests_hybrid

clean:
	rm -f build/poisson_seq build/poisson_omp build/poisson_mpi build/poisson_hybrid build/tests build/tests_omp build/tests_mpi build/tests_hybrid
