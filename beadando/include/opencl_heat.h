#ifndef OPENCL_HEAT_H
#define OPENCL_HEAT_H

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include "grid.h"

/*
 * OpenCL module
 *
 * Handles GPU-based computation of the heat simulation.
 * Responsible for initialization, memory management,
 * kernel execution, and cleanup.
 */

/* Initializes the OpenCL environment and allocates device resources. */
void init_opencl(const Grid *g);

/* Uploads the current simulation state from host memory to the device. */
void upload_state_to_device(const Grid *g);

/*
 * Executes one simulation step entirely on the GPU.
 * Uses buffer swapping to avoid unnecessary memory transfers.
 */
void run_kernel_step_device_only(const Grid *g);

/* Downloads the current simulation state from the device to host memory. */
void download_state_from_device(Grid *g);

/*
 * Performs a full simulation step with host-device synchronization.
 * Useful for rendering, but less efficient for benchmarking.
 */
void run_kernel(Grid *g);

/* Waits until all queued OpenCL commands have completed. */
void finish_opencl(void);

/* Releases all OpenCL resources. */
void cleanup_opencl(void);

#endif