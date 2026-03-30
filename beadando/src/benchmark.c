#include <stdio.h>
#include <stdlib.h>

#include "benchmark.h"
#include "grid.h"
#include "util.h"
#include "opencl_heat.h"

/*
 * Executes one CPU simulation step.
 *
 * Each cell is updated from its four direct neighbors.
 * Border cells are kept at 0.0, while cells marked in source_map
 * remain permanent heat sources with temperature 1.0.
 */
static void run_cpu_step(float *current, float *next, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;

            /* Keep the boundary cells cold. */
            if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
            {
                next[idx] = 0.0f;
                continue;
            }

            /* Keep permanent heat sources at maximum temperature. */
            if (source_map[idx])
            {
                next[idx] = 1.0f;
                continue;
            }

            /* Read the four direct neighbors. */
            float up = current[(y - 1) * width + x];
            float down = current[(y + 1) * width + x];
            float left = current[y * width + (x - 1)];
            float right = current[y * width + (x + 1)];

            /* Compute the new temperature as the average of neighbors. */
            next[idx] = 0.25f * (up + down + left + right);
        }
    }
}

/*
 * Runs the CPU version of the simulation for a given number of iterations.
 *
 * Two buffers are used and swapped after each step to avoid overwriting
 * data that is still needed for the current computation.
 */
static void run_cpu_simulation(float *a, float *b, int width, int height, int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        run_cpu_step(a, b, width, height);

        /* Swap current and next buffers. */
        float *temp = a;
        a = b;
        b = temp;
    }
}

/*
 * Measures the execution time of the CPU implementation.
 *
 * A fresh local grid is allocated for the benchmark to ensure that
 * each measurement starts from the same initial state.
 */
double benchmark_cpu(int iterations)
{
    float *a = (float *)malloc(sizeof(float) * sim_width * sim_height);
    float *b = (float *)malloc(sizeof(float) * sim_width * sim_height);

    if (a == NULL || b == NULL)
    {
        printf("CPU benchmark memory allocation failed.\n");
        free(a);
        free(b);
        return -1.0;
    }

    /* Initialize both local buffers with zero temperature. */
    for (int i = 0; i < sim_width * sim_height; i++)
    {
        a[i] = 0.0f;
        b[i] = 0.0f;
    }

    /* Place the initial heat source in the center. */
    int cx = sim_width / 2;
    int cy = sim_height / 2;

    for (int y = cy - 10; y <= cy + 10; y++)
    {
        for (int x = cx - 10; x <= cx + 10; x++)
        {
            int idx = y * sim_width + x;
            a[idx] = 1.0f;
            b[idx] = 1.0f;
        }
    }

    double start = get_time_ms();

    run_cpu_simulation(a, b, sim_width, sim_height, iterations);

    double end = get_time_ms();

    free(a);
    free(b);

    return end - start;
}

/*
 * Measures the execution time of the OpenCL implementation.
 *
 * The simulation state is reset before the measurement, then uploaded
 * to the device. The kernel is executed multiple times without
 * unnecessary host-device transfers between iterations.
 */
double benchmark_opencl(int iterations)
{
    reset_grid();
    upload_state_to_device();

    double start = get_time_ms();

    for (int i = 0; i < iterations; i++)
    {
        run_kernel_step_device_only();
    }

    /* Ensure that all queued OpenCL operations are finished. */
    clFinish(queue);

    double end = get_time_ms();

    /* Download the final state back to host memory. */
    download_state_from_device();

    return end - start;
}

/*
 * Creates the CSV file and writes its header if the file does not exist yet.
 */
void ensure_csv_header(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (f != NULL)
    {
        fclose(f);
        return;
    }

    f = fopen(filename, "w");
    if (f == NULL)
    {
        printf("Failed to create CSV file: %s\n", filename);
        return;
    }

    fprintf(f, "width,height,iterations,cpu_time_ms,opencl_time_ms,speedup\n");
    fclose(f);
}

/*
 * Appends one benchmark result row to the CSV file.
 *
 * Speedup is computed as CPU time divided by OpenCL time.
 */
void save_benchmark_csv(const char *filename,
                        int width,
                        int height,
                        int iterations,
                        double cpu_time,
                        double opencl_time)
{
    FILE *f = fopen(filename, "a");

    if (f == NULL)
    {
        printf("Failed to open CSV file: %s\n", filename);
        return;
    }

    double speedup = 0.0;
    if (opencl_time > 0.0)
    {
        speedup = cpu_time / opencl_time;
    }

    fprintf(f, "%d,%d,%d,%.3f,%.3f,%.3f\n",
            width, height, iterations, cpu_time, opencl_time, speedup);

    fclose(f);
}

/*
 * Runs a simple automated benchmark on the current simulation size
 * using multiple iteration counts.
 */
void run_automated_benchmarks(const char *filename)
{
    int test_iterations[] = {100, 500, 1000, 2000};
    int test_count = sizeof(test_iterations) / sizeof(test_iterations[0]);

    ensure_csv_header(filename);

    printf("\n=== Automated benchmark started ===\n");

    for (int i = 0; i < test_count; i++)
    {
        int iterations = test_iterations[i];

        printf("\n[Benchmark] Iterations: %d\n", iterations);

        double cpu_time = benchmark_cpu(iterations);
        double opencl_time = benchmark_opencl(iterations);

        printf("CPU time:    %.3f ms\n", cpu_time);
        printf("OpenCL time: %.3f ms\n", opencl_time);

        if (opencl_time > 0.0)
        {
            printf("Speedup:     %.2fx\n", cpu_time / opencl_time);
        }

        save_benchmark_csv(filename,
                           sim_width, sim_height, iterations, cpu_time, opencl_time);
    }

    printf("\n=== Automated benchmark finished ===\n");
}

/*
 * Runs averaged benchmarks for multiple grid sizes.
 *
 * For each size, the simulation grid and OpenCL environment are reinitialized,
 * then both CPU and OpenCL versions are measured multiple times and averaged.
 */
void run_multi_size_benchmarks(const char *filename)
{
    int sizes[] = {256, 512, 1024};
    int iterations = 500;
    int runs = 100;

    int count = sizeof(sizes) / sizeof(sizes[0]);

    ensure_csv_header(filename);

    printf("\n=== Averaged multi-size benchmark started ===\n");

    for (int i = 0; i < count; i++)
    {
        int size = sizes[i];

        printf("\n[Benchmark] %dx%d, iter=%d, runs=%d\n",
               size, size, iterations, runs);

        /* Recreate the simulation state for the current test size. */
        free_grid();
        cleanup_opencl();

        init_grid(size, size, 1);
        init_opencl();

        double cpu_time = benchmark_cpu_avg(iterations, runs);
        double opencl_time = benchmark_opencl_avg(iterations, runs);

        printf("CPU avg:    %.3f ms\n", cpu_time);
        printf("OpenCL avg: %.3f ms\n", opencl_time);

        if (opencl_time > 0.0)
        {
            printf("Speedup:    %.2fx\n", cpu_time / opencl_time);
        }

        save_benchmark_csv(filename,
                           sim_width, sim_height,
                           iterations,
                           cpu_time, opencl_time);
    }

    printf("\n=== Benchmark finished ===\n");
}

/*
 * Computes the average CPU benchmark time over multiple runs.
 */
double benchmark_cpu_avg(int iterations, int runs)
{
    double sum = 0.0;

    for (int i = 0; i < runs; i++)
    {
        double t = benchmark_cpu(iterations);
        sum += t;
    }

    return sum / runs;
}

/*
 * Computes the average OpenCL benchmark time over multiple runs.
 */
double benchmark_opencl_avg(int iterations, int runs)
{
    double sum = 0.0;

    for (int i = 0; i < runs; i++)
    {
        double t = benchmark_opencl(iterations);
        sum += t;
    }

    return sum / runs;
}