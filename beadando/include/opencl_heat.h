#ifndef OPENCL_HEAT_H
#define OPENCL_HEAT_H

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

/*
 * OpenCL module
 *
 * Handles GPU-based computation of the heat simulation.
 * Responsible for initialization, memory management,
 * kernel execution, and cleanup.
 */

/* Core OpenCL objects. */
extern cl_context context;
extern cl_command_queue queue;
extern cl_program program;
extern cl_kernel kernel;

/*
 * Device memory buffers:
 *  - d_current: current temperature grid
 *  - d_next: next temperature grid
 *  - d_source_map: permanent heat source flags
 */
extern cl_mem d_current;
extern cl_mem d_next;
extern cl_mem d_source_map;

/*
 * Active buffers used during iterative computation.
 * These are swapped after each simulation step.
 */
extern cl_mem d_current_active;
extern cl_mem d_next_active;

/* Initializes the OpenCL environment and allocates device resources. */
void init_opencl(void);

/* Uploads the current simulation state from host memory to the device. */
void upload_state_to_device(void);

/*
 * Executes one simulation step entirely on the GPU.
 * Uses buffer swapping to avoid unnecessary memory transfers.
 */
void run_kernel_step_device_only(void);

/* Downloads the current simulation state from the device to host memory. */
void download_state_from_device(void);

/*
 * Performs a full simulation step with host-device synchronization.
 * Useful for rendering, but less efficient for benchmarking.
 */
void run_kernel(void);

/* Releases all OpenCL resources. */
void cleanup_opencl(void);

#endif