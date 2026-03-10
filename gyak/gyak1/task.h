#ifndef TASKS_H
#define TASKS_H

#include "ocl_utils.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 3) mapping
void task_map_index(const OclContext* ocl, int use_local_index, int n);
void task_map_reverse(const OclContext* ocl, int n);
void task_map_swap_neighbors(const OclContext* ocl, int n);

// 4) vektorösszeadás (OpenCL elrejtve)
int  vec_add(float* out, const float* a, const float* b, size_t n); // "normál" API
void task_vec_add_demo(const OclContext* ocl, int n);

// 5) hiányzó elemek pótlása
void make_missing_input(uint32_t* a, uint8_t* missingMask, size_t n, unsigned seed, int holes);
void task_fill_missing(const OclContext* ocl, int n);

// 6) rang
void task_rank(const OclContext* ocl, int n);

// 7) előfordulásszám + egyediség
void task_occurrence(const OclContext* ocl, int n);

// 8) min/max (redukció)
void task_minmax(const OclContext* ocl, int n);

// 9) csúszóátlag
void task_sliding_avg(const OclContext* ocl, int n, int radius);

// 10) prím vizsgálat
void task_prime_test(const OclContext* ocl, uint32_t x);

#ifdef __cplusplus
}
#endif

#endif