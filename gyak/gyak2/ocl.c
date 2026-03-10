#include "ocl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* cl_error_name(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        default: return "CL_UNKNOWN_ERROR";
    }
}

static void die_cl(const char* msg, cl_int err) {
    fprintf(stderr, "%s: %d (%s)\n", msg, err, cl_error_name(err));
    exit(1);
}

static int pick_device(cl_platform_id* outP, cl_device_id* outD) {
    cl_uint pc = 0;
    if (clGetPlatformIDs(0, NULL, &pc) != CL_SUCCESS || pc == 0) return 0;
    cl_platform_id* ps = (cl_platform_id*)calloc(pc, sizeof(*ps));
    clGetPlatformIDs(pc, ps, NULL);

    // GPU first
    for (cl_uint i=0;i<pc;i++){
        cl_uint dc=0;
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 0, NULL, &dc)==CL_SUCCESS && dc>0){
            cl_device_id d;
            clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 1, &d, NULL);
            *outP=ps[i]; *outD=d; free(ps); return 1;
        }
    }
    // CPU fallback
    for (cl_uint i=0;i<pc;i++){
        cl_uint dc=0;
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_CPU, 0, NULL, &dc)==CL_SUCCESS && dc>0){
            cl_device_id d;
            clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_CPU, 1, &d, NULL);
            *outP=ps[i]; *outD=d; free(ps); return 1;
        }
    }
    free(ps);
    return 0;
}

int ocl_init(Ocl* ocl) {
    memset(ocl, 0, sizeof(*ocl));
    if (!pick_device(&ocl->platform, &ocl->device)) return 0;

    cl_int err=0;
    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return 0;

    // PROFILING ENABLED queue for task #3/#5
    ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) return 0;

    return 1;
}

void ocl_cleanup(Ocl* ocl) {
    if (ocl->queue) clReleaseCommandQueue(ocl->queue);
    if (ocl->context) clReleaseContext(ocl->context);
    memset(ocl, 0, sizeof(*ocl));
}

char* read_text_file(const char* path, size_t* outSz) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }

    char* s = (char*)malloc((size_t)sz + 1);
    if (!s) { fclose(f); return NULL; }
    size_t rd = fread(s, 1, (size_t)sz, f);
    fclose(f);
    s[rd] = 0;
    if (outSz) *outSz = rd;
    return s;
}

static char* get_build_log(cl_program p, cl_device_id d) {
    size_t sz=0;
    clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    if (sz == 0) return NULL;
    char* log = (char*)malloc(sz + 1);
    if (!log) return NULL;
    clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, sz, log, NULL);
    log[sz] = 0;
    return log;
}

BuildResult build_program_from_file(const Ocl* ocl, const char* path, const char* options) {
    BuildResult br;
    memset(&br, 0, sizeof(br));

    size_t sz=0;
    char* src = read_text_file(path, &sz);
    if (!src) {
        fprintf(stderr, "Cannot read kernel file: %s\n", path);
        br.build_err = CL_INVALID_VALUE;
        return br;
    }

    cl_int err=0;
    const char* arr[1] = { src };
    br.program = clCreateProgramWithSource(ocl->context, 1, arr, NULL, &err);
    free(src);
    if (err != CL_SUCCESS) {
        br.build_err = err;
        return br;
    }

    err = clBuildProgram(br.program, 1, &ocl->device, options, NULL, NULL);
    br.build_err = err;
    br.build_log = get_build_log(br.program, ocl->device);
    return br;
}

void free_build_result(BuildResult* br) {
    if (!br) return;
    if (br->build_log) free(br->build_log);
    if (br->program) clReleaseProgram(br->program);
    memset(br, 0, sizeof(*br));
}

cl_kernel create_kernel_or_die(cl_program p, const char* name) {
    cl_int err=0;
    cl_kernel k = clCreateKernel(p, name, &err);
    if (err != CL_SUCCESS) die_cl("clCreateKernel failed", err);
    return k;
}