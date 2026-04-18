#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include "grid.h"
#include "opencl_heat.h"

/* OpenCL objects used internally by the module. */
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;

static cl_mem d_current = NULL;
static cl_mem d_next = NULL;
static cl_mem d_source_map = NULL;

static cl_mem d_current_active = NULL;
static cl_mem d_next_active = NULL;

/*
 * Loads the OpenCL kernel source code from a file into a dynamically
 * allocated null-terminated string.
 *
 * Returns:
 *   Pointer to the loaded source code on success,
 *   NULL on failure.
 */
static char *load_kernel_source(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (f == NULL)
    {
        printf("Error: failed to open kernel file: %s\n", filename);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0)
    {
        fclose(f);
        printf("Error: failed to seek to the end of the kernel file.\n");
        return NULL;
    }

    long size = ftell(f);
    if (size < 0)
    {
        fclose(f);
        printf("Error: failed to determine kernel file size.\n");
        return NULL;
    }

    rewind(f);

    char *source = (char *)malloc((size_t)size + 1);
    if (source == NULL)
    {
        fclose(f);
        printf("Error: memory allocation failed for kernel source.\n");
        return NULL;
    }

    size_t read_size = fread(source, 1, (size_t)size, f);
    source[read_size] = '\0';

    fclose(f);
    return source;
}

/*
 * Prints the OpenCL program build log.
 *
 * This is especially useful when kernel compilation fails,
 * because it provides detailed compiler messages from the OpenCL driver.
 */
static void print_build_log(cl_program prog, cl_device_id device)
{
    size_t log_size = 0;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    if (log_size > 1)
    {
        char *log = (char *)malloc(log_size);
        if (log != NULL)
        {
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("OpenCL build log:\n%s\n", log);
            free(log);
        }
    }
}

/*
 * Initializes the OpenCL environment.
 *
 * The function:
 * - selects a platform and GPU device,
 * - creates a context and command queue,
 * - loads and builds the kernel program,
 * - creates the device buffers needed by the simulation.
 */
void init_opencl(const Grid *g)
{
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;

    /* Find the first available OpenCL platform. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS || platform == NULL)
    {
        printf("Error: failed to find an OpenCL platform.\n");
        exit(1);
    }

    /* Select the first available GPU device on the platform. */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS || device == NULL)
    {
        printf("Error: failed to find a GPU device.\n");
        exit(1);
    }

    /* Create an OpenCL context. */
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || context == NULL)
    {
        printf("Error: clCreateContext failed.\n");
        exit(1);
    }

    /* Create a command queue used to submit work to the device. */
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS || queue == NULL)
    {
        printf("Error: clCreateCommandQueue failed.\n");
        exit(1);
    }

    /* Load kernel source code from file. */
    char *source = load_kernel_source("kernels/heat_kernel.cl");
    if (source == NULL)
    {
        exit(1);
    }

    /* Create and build the OpenCL program. */
    program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
    if (err != CL_SUCCESS || program == NULL)
    {
        printf("Error: clCreateProgramWithSource failed.\n");
        free(source);
        exit(1);
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: clBuildProgram failed.\n");
        print_build_log(program, device);
        free(source);
        exit(1);
    }

    /* Create the kernel object from the compiled program. */
    kernel = clCreateKernel(program, "heat_step", &err);
    if (err != CL_SUCCESS || kernel == NULL)
    {
        printf("Error: clCreateKernel failed.\n");
        free(source);
        exit(1);
    }

    /* Allocate device buffer for the current temperature grid. */
    d_current = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               sizeof(float) * g->width * g->height,
                               NULL,
                               &err);
    if (err != CL_SUCCESS || d_current == NULL)
    {
        printf("Error: failed to create d_current buffer.\n");
        free(source);
        exit(1);
    }

    /* Allocate device buffer for the next temperature grid. */
    d_next = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            sizeof(float) * g->width * g->height,
                            NULL,
                            &err);
    if (err != CL_SUCCESS || d_next == NULL)
    {
        printf("Error: failed to create d_next buffer.\n");
        free(source);
        exit(1);
    }

    /* Allocate device buffer for the permanent heat source map. */
    d_source_map = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  sizeof(unsigned char) * g->width * g->height,
                                  NULL,
                                  &err);
    if (err != CL_SUCCESS || d_source_map == NULL)
    {
        printf("Error: failed to create d_source_map buffer.\n");
        free(source);
        exit(1);
    }

    /* Initially, the active buffers are the original device buffers. */
    d_current_active = d_current;
    d_next_active = d_next;

    free(source);
}

/*
 * Uploads the current host-side simulation state to the device.
 *
 * This includes:
 * - current temperature grid
 * - next temperature grid
 * - permanent heat source map
 */
void upload_state_to_device(const Grid *g)
{
    clEnqueueWriteBuffer(queue,
                         d_current,
                         CL_TRUE,
                         0,
                         sizeof(float) * g->width * g->height,
                         g->current,
                         0,
                         NULL,
                         NULL);

    clEnqueueWriteBuffer(queue,
                         d_next,
                         CL_TRUE,
                         0,
                         sizeof(float) * g->width * g->height,
                         g->next,
                         0,
                         NULL,
                         NULL);

    clEnqueueWriteBuffer(queue,
                         d_source_map,
                         CL_TRUE,
                         0,
                         sizeof(unsigned char) * g->width * g->height,
                         g->source_map,
                         0,
                         NULL,
                         NULL);

    /* Reset active device buffers before starting a new simulation phase. */
    d_current_active = d_current;
    d_next_active = d_next;
}

/*
 * Executes one OpenCL simulation step entirely on the device.
 *
 * After kernel execution, the active input and output buffers are swapped,
 * so the next iteration can continue without extra host-device copies.
 */
void run_kernel_step_device_only(const Grid *g)
{
    cl_int err;

    /*
     * Try a fixed 16x16 local work-group size.
     * This is a common choice for 2D grid-based kernels.
     */
    size_t local[2] = {16, 16};

    /*
     * The global size must be a multiple of the local size,
     * so round up both dimensions if necessary.
     */
    size_t global[2];
    global[0] = ((size_t)g->width + local[0] - 1) / local[0] * local[0];
    global[1] = ((size_t)g->height + local[1] - 1) / local[1] * local[1];

    int w = g->width;
    int h = g->height;

    /* Pass all required kernel arguments. */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_current_active);
    if (err != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg(0) failed.\n");
        exit(1);
    }

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_next_active);
    if (err != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg(1) failed.\n");
        exit(1);
    }

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_source_map);
    if (err != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg(2) failed.\n");
        exit(1);
    }

    err = clSetKernelArg(kernel, 3, sizeof(int), &w);
    if (err != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg(3) failed.\n");
        exit(1);
    }

    err = clSetKernelArg(kernel, 4, sizeof(int), &h);
    if (err != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg(4) failed.\n");
        exit(1);
    }

    /*
     * Launch the kernel over the padded global range
     * using the selected local work-group size.
     */
    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 2,
                                 NULL,
                                 global,
                                 local,
                                 0,
                                 NULL,
                                 NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel failed (%d).\n", err);
        exit(1);
    }

    clFinish(queue);

    /* Swap active device buffers after the step. */
    cl_mem temp = d_current_active;
    d_current_active = d_next_active;
    d_next_active = temp;
}

/*
 * Downloads the current device state back to the host-side grid.
 *
 * The most recent state is always stored in d_current_active.
 */
void download_state_from_device(Grid *g)
{
    clEnqueueReadBuffer(queue,
                        d_current_active,
                        CL_TRUE,
                        0,
                        sizeof(float) * g->width * g->height,
                        g->current,
                        0,
                        NULL,
                        NULL);
}

/*
 * Executes one complete simulation step with host-device synchronization.
 *
 * This version is mainly useful for interactive rendering, where the updated
 * grid needs to be displayed after each iteration.
 */
void run_kernel(Grid *g)
{
    upload_state_to_device(g);
    run_kernel_step_device_only(g);
    download_state_from_device(g);
}

/*
 * Blocks until all queued OpenCL commands have completed.
 */
void finish_opencl(void)
{
    if (queue != NULL)
    {
        clFinish(queue);
    }
}

/*
 * Releases all OpenCL resources used by the simulation.
 */
void cleanup_opencl(void)
{
    if (d_current != NULL)
        clReleaseMemObject(d_current);
    if (d_next != NULL)
        clReleaseMemObject(d_next);
    if (d_source_map != NULL)
        clReleaseMemObject(d_source_map);
    if (kernel != NULL)
        clReleaseKernel(kernel);
    if (program != NULL)
        clReleaseProgram(program);
    if (queue != NULL)
        clReleaseCommandQueue(queue);
    if (context != NULL)
        clReleaseContext(context);

    d_current = NULL;
    d_next = NULL;
    d_source_map = NULL;
    d_current_active = NULL;
    d_next_active = NULL;
    kernel = NULL;
    program = NULL;
    queue = NULL;
    context = NULL;
}