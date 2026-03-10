#ifndef OCL_COMMON_H
#define OCL_COMMON_H

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue; // profiling enabled
} Ocl;

const char* cl_error_name(cl_int e);

int  ocl_init(Ocl* ocl);
void ocl_cleanup(Ocl* ocl);

char* read_text_file(const char* path, size_t* outSz);

typedef struct {
    cl_program program;
    char* build_log;   // malloc, can be NULL
    cl_int err;
} BuildResult;

BuildResult build_program_from_file(const Ocl* ocl, const char* path, const char* options);
void free_build_result(BuildResult* br);

cl_kernel kernel_or_die(cl_program p, const char* name);

double event_ms(cl_event ev); // profiling: ns -> ms

#ifdef __cplusplus
}
#endif

#endif