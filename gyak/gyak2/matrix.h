#ifndef MATRIX_H
#define MATRIX_H

#include "ocl.h"

void mat_transpose_demo(const Ocl* ocl, int R, int C);
void mat_mul_demo_naive(const Ocl* ocl, int R, int K, int C);
void mat_mul_demo_tiled(const Ocl* ocl, int R, int K, int C, int tile);
void mat_row_sum_demo(const Ocl* ocl, int R, int C);
void mat_col_sum_demo(const Ocl* ocl, int R, int C);

void benchmark_matrix(const Ocl* ocl, int Nmin, int Nmax, int step);

#endif //MATRIX_H