#ifndef BENCHMARK_H
#define BENCHMARK_H

/*
 * Benchmark module
 *
 * Provides functions to measure and compare the performance
 * of CPU and OpenCL implementations of the heat simulation.
 */

/* Measures execution time of the CPU implementation. */
double benchmark_cpu(int iterations);

/* Measures execution time of the OpenCL implementation. */
double benchmark_opencl(int iterations);

/*
 * Ensures that the CSV file exists and contains a header.
 * If the file does not exist, it will be created.
 */
void ensure_csv_header(const char *filename);

/*
 * Saves one benchmark result into the CSV file.
 *
 * Parameters:
 *  - width, height: simulation grid size
 *  - iterations: number of simulation steps
 *  - cpu_time: CPU execution time (ms)
 *  - opencl_time: OpenCL execution time (ms)
 */
void save_benchmark_csv(const char *filename,
                        int width,
                        int height,
                        int iterations,
                        double cpu_time,
                        double opencl_time);

/*
 * Runs benchmarks with multiple iteration counts
 * on the current grid size.
 */
void run_automated_benchmarks(const char *filename);

/*
 * Runs benchmarks for multiple grid sizes.
 * The simulation is reinitialized for each size.
 */
void run_multi_size_benchmarks(const char *filename);

/*
 * Measures average CPU execution time over multiple runs.
 */
double benchmark_cpu_avg(int iterations, int runs);

/*
 * Measures average OpenCL execution time over multiple runs.
 */
double benchmark_opencl_avg(int iterations, int runs);

#endif