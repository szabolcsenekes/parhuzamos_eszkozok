#include <stdio.h>
#include <stdlib.h>

#include "benchmark.h"
#include "grid.h"
#include "util.h"
#include "opencl_heat.h"

/* ============================================================
 * CPU IMPLEMENTATION
 * ============================================================
 */

/*
 * Executes a single CPU simulation step.
 *
 * Each grid cell is updated based on the average of its
 * four direct neighbors (up, down, left, right).
 *
 * Boundary cells are fixed at 0.0 temperature.
 * Cells marked in source_map remain constant heat sources.
 */
static void run_cpu_step(const float *current,
                         float *next,
                         const unsigned char *source_map,
                         int width,
                         int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;

            /* Boundary condition: keep edges cold. */
            if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
            {
                next[idx] = 0.0f;
                continue;
            }

            /* Preserve permanent heat sources. */
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

            /* Compute the average temperature. */
            next[idx] = 0.25f * (up + down + left + right);
        }
    }
}

/*
 * Runs the CPU simulation for a given number of iterations.
 *
 * Uses double buffering (pointer swapping) to avoid overwriting
 * data needed for the current iteration.
 */
static void run_cpu_simulation(float *a,
                               float *b,
                               const unsigned char *source_map,
                               int width,
                               int height,
                               int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        run_cpu_step(a, b, source_map, width, height);

        /* Swap current and next buffers. */
        float *temp = a;
        a = b;
        b = temp;
    }
}

/*
 * Measures CPU execution time for the simulation.
 *
 * A fresh local grid is allocated for each measurement to ensure
 * consistent starting conditions.
 */
double benchmark_cpu(Grid *g, int iterations)
{
    int size = g->width * g->height;

    float *a = (float *)malloc(sizeof(float) * size);
    float *b = (float *)malloc(sizeof(float) * size);

    if (a == NULL || b == NULL)
    {
        printf("CPU benchmark memory allocation failed.\n");
        free(a);
        free(b);
        return -1.0;
    }

    /* Initialize both local grids with zero temperature. */
    for (int i = 0; i < size; i++)
    {
        a[i] = 0.0f;
        b[i] = 0.0f;
    }

    /* Create an initial heat source in the center. */
    int cx = g->width / 2;
    int cy = g->height / 2;

    for (int y = cy - 10; y <= cy + 10; y++)
    {
        for (int x = cx - 10; x <= cx + 10; x++)
        {
            if (x >= 0 && x < g->width && y >= 0 && y < g->height)
            {
                int idx = y * g->width + x;
                a[idx] = 1.0f;
                b[idx] = 1.0f;
            }
        }
    }

    double start = get_time_ms();

    run_cpu_simulation(a, b, g->source_map, g->width, g->height, iterations);

    double end = get_time_ms();

    free(a);
    free(b);

    return end - start;
}

/* ============================================================
 * OPENCL IMPLEMENTATION (DETAILED BREAKDOWN)
 * ============================================================
 */

/*
 * Measures OpenCL execution time with phase breakdown.
 *
 * The simulation is divided into three main phases:
 * 1. Upload (host -> device transfer)
 * 2. Compute (kernel execution)
 * 3. Download (device -> host transfer)
 *
 * This helps identify performance bottlenecks such as
 * memory transfer overhead.
 */
OpenCLBreakdown benchmark_opencl_breakdown(Grid *g, int iterations)
{
    OpenCLBreakdown result = {0};

    /* Reset the simulation to a known initial state. */
    reset_grid(g);

    /* --- Upload phase --- */
    double t0 = get_time_ms();
    upload_state_to_device(g);
    double t1 = get_time_ms();

    /* --- Compute phase --- */
    for (int i = 0; i < iterations; i++)
    {
        run_kernel_step_device_only(g);
    }

    finish_opencl();
    double t2 = get_time_ms();

    /* --- Download phase --- */
    download_state_from_device(g);
    double t3 = get_time_ms();

    result.upload_ms = t1 - t0;
    result.compute_ms = t2 - t1;
    result.download_ms = t3 - t2;
    result.total_ms = t3 - t0;

    return result;
}

/*
 * Compatibility wrapper: returns total OpenCL time only.
 */
double benchmark_opencl(Grid *g, int iterations)
{
    OpenCLBreakdown result = benchmark_opencl_breakdown(g, iterations);
    return result.total_ms;
}

/* ============================================================
 * AVERAGING FUNCTIONS
 * ============================================================
 */

/*
 * Computes average CPU execution time over multiple runs.
 */
double benchmark_cpu_avg(Grid *g, int iterations, int runs)
{
    double sum = 0.0;

    for (int i = 0; i < runs; i++)
    {
        sum += benchmark_cpu(g, iterations);
    }

    return sum / runs;
}

/*
 * Computes average OpenCL execution time over multiple runs.
 */
double benchmark_opencl_avg(Grid *g, int iterations, int runs)
{
    double sum = 0.0;

    for (int i = 0; i < runs; i++)
    {
        sum += benchmark_opencl(g, iterations);
    }

    return sum / runs;
}

/*
 * Computes average OpenCL phase breakdown over multiple runs.
 */
OpenCLBreakdown benchmark_opencl_breakdown_avg(Grid *g, int iterations, int runs)
{
    OpenCLBreakdown avg = {0};

    for (int i = 0; i < runs; i++)
    {
        OpenCLBreakdown t = benchmark_opencl_breakdown(g, iterations);

        avg.upload_ms += t.upload_ms;
        avg.compute_ms += t.compute_ms;
        avg.download_ms += t.download_ms;
        avg.total_ms += t.total_ms;
    }

    avg.upload_ms /= runs;
    avg.compute_ms /= runs;
    avg.download_ms /= runs;
    avg.total_ms /= runs;

    return avg;
}

/* ============================================================
 * CSV OUTPUT
 * ============================================================
 */

/*
 * Ensures that the CSV file exists and contains a header row.
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

    fprintf(f,
            "width,height,iterations,cpu_time_ms,"
            "upload_ms,compute_ms,download_ms,total_ms,"
            "speedup_total,speedup_compute\n");

    fclose(f);
}

/*
 * Appends one benchmark result with total OpenCL time.
 *
 * This helper keeps compatibility with the old API.
 */
void save_benchmark_csv(const char *filename,
                        int width,
                        int height,
                        int iterations,
                        double cpu_time,
                        double opencl_time)
{
    OpenCLBreakdown ocl;
    ocl.upload_ms = 0.0;
    ocl.compute_ms = opencl_time;
    ocl.download_ms = 0.0;
    ocl.total_ms = opencl_time;

    save_benchmark_csv_detailed(filename,
                                width,
                                height,
                                iterations,
                                cpu_time,
                                ocl);
}

/*
 * Appends one benchmark result with detailed OpenCL breakdown
 * to the CSV file.
 */
void save_benchmark_csv_detailed(const char *filename,
                                 int width,
                                 int height,
                                 int iterations,
                                 double cpu_time,
                                 OpenCLBreakdown ocl)
{
    FILE *f = fopen(filename, "a");
    if (f == NULL)
    {
        printf("Failed to open CSV file: %s\n", filename);
        return;
    }

    double speedup_total = 0.0;
    double speedup_compute = 0.0;

    if (ocl.total_ms > 0.0)
    {
        speedup_total = cpu_time / ocl.total_ms;
    }

    if (ocl.compute_ms > 0.0)
    {
        speedup_compute = cpu_time / ocl.compute_ms;
    }

    fprintf(f, "%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            width,
            height,
            iterations,
            cpu_time,
            ocl.upload_ms,
            ocl.compute_ms,
            ocl.download_ms,
            ocl.total_ms,
            speedup_total,
            speedup_compute);

    fclose(f);
}

/* ============================================================
 * BENCHMARK RUNNER
 * ============================================================
 */

/*
 * Runs averaged benchmarks for multiple grid sizes.
 *
 * For each size:
 * - reinitializes the simulation grid
 * - reinitializes the OpenCL environment
 * - measures CPU and OpenCL performance
 * - saves results to CSV
 */
void run_multi_size_benchmarks(const char *filename)
{
    int sizes[] = {64, 128, 256, 512, 1024};
    int iterations = 500;
    int runs = 50;

    int count = (int)(sizeof(sizes) / sizeof(sizes[0]));

    Grid grid_state = {0};

    ensure_csv_header(filename);

    printf("\n=== Averaged multi-size benchmark started ===\n");

    for (int i = 0; i < count; i++)
    {
        int size = sizes[i];

        printf("\n[Benchmark] %dx%d, iter=%d, runs=%d\n",
               size, size, iterations, runs);

        init_grid(&grid_state, size, size, 1);
        init_opencl(&grid_state);

        double cpu_time = benchmark_cpu_avg(&grid_state, iterations, runs);
        OpenCLBreakdown ocl = benchmark_opencl_breakdown_avg(&grid_state, iterations, runs);

        printf("CPU avg:         %.3f ms\n", cpu_time);
        printf("OpenCL upload:   %.3f ms\n", ocl.upload_ms);
        printf("OpenCL compute:  %.3f ms\n", ocl.compute_ms);
        printf("OpenCL download: %.3f ms\n", ocl.download_ms);
        printf("OpenCL total:    %.3f ms\n", ocl.total_ms);

        if (ocl.total_ms > 0.0)
        {
            printf("Speedup total:   %.2fx\n", cpu_time / ocl.total_ms);
        }

        if (ocl.compute_ms > 0.0)
        {
            printf("Speedup compute: %.2fx\n", cpu_time / ocl.compute_ms);
        }

        save_benchmark_csv_detailed(filename,
                                    grid_state.width,
                                    grid_state.height,
                                    iterations,
                                    cpu_time,
                                    ocl);

        cleanup_opencl();
        free_grid(&grid_state);
    }

    printf("\n=== Benchmark finished ===\n");
}