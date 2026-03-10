#ifndef OCL_UTILS_H
#define OCL_UTILS_H

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
    cl_command_queue queue;
} OclContext;

const char* ocl_errstr(cl_int err);

void ocl_print_device_info_like_clinfo(void);

int  ocl_init_default(OclContext* ocl);   // GPU preferált, ha nincs akkor CPU
void ocl_cleanup(OclContext* ocl);

char* ocl_read_text_file(const char* path, size_t* out_size);

cl_program ocl_build_program_from_file(
    const OclContext* ocl,
    const char* file_path,
    const char* build_options
);

cl_kernel ocl_create_kernel(cl_program program, const char* kernel_name);

cl_mem ocl_create_buffer(cl_context ctx, cl_mem_flags flags, size_t bytes);

cl_int ocl_finish(cl_command_queue q);

#ifdef __cplusplus
}
#endif

#endif //OCL_UTILS_H