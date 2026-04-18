#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "grid.h"

/*
 * Benchmark module
 *
 * Provides functions to measure and compare the performance
 * of CPU and OpenCL implementations of the heat simulation.
 */

typedef struct OpenCLBreakdown
{
    double upload_ms;
    double compute_ms;
    double download_ms;
    double total_ms;
} OpenCLBreakdown;

/* Measures execution time of the CPU implementation. */
double benchmark_cpu(Grid *g, int iterations);

/* Measures execution time of the OpenCL implementation. */
double benchmark_opencl(Grid *g, int iterations);

/* Measures OpenCL execution time with detailed phase breakdown. */
OpenCLBreakdown benchmark_opencl_breakdown(Grid *g, int iterations);

/*
 * Ensures that the CSV file exists and contains a header.
 * If the file does not exist, it will be created.
 */
void ensure_csv_header(const char *filename);

/*
 * Saves one benchmark result into the CSV file.
 */
void save_benchmark_csv(const char *filename,
                        int width,
                        int height,
                        int iterations,
                        double cpu_time,
                        double opencl_time);

/*
 * Saves one benchmark result with OpenCL phase breakdown.
 */
void save_benchmark_csv_detailed(const char *filename,
                                 int width,
                                 int height,
                                 int iterations,
                                 double cpu_time,
                                 OpenCLBreakdown ocl);

/*
 * Runs benchmarks for multiple grid sizes.
 * The simulation is reinitialized for each size.
 */
void run_multi_size_benchmarks(const char *filename);

/*
 * Measures average CPU execution time over multiple runs.
 */
double benchmark_cpu_avg(Grid *g, int iterations, int runs);

/*
 * Measures average OpenCL execution time over multiple runs.
 */
double benchmark_opencl_avg(Grid *g, int iterations, int runs);

/*
 * Measures average OpenCL breakdown over multiple runs.
 */
OpenCLBreakdown benchmark_opencl_breakdown_avg(Grid *g, int iterations, int runs);

#endif