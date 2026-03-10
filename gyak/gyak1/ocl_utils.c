#include "ocl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OCL_CHECK(e) do { \
    cl_int _err = (e); \
    if (_err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d (%s) at %s:%d\n", _err, ocl_errstr(_err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

const char* ocl_errstr(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
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
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        default: return "CL_UNKNOWN_ERROR";
    }
}

static void print_platform_info(cl_platform_id p) {
    char buf[4096];
    size_t n = 0;
    OCL_CHECK(clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(buf), buf, &n));
    printf("Platform: %s\n", buf);

    OCL_CHECK(clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(buf), buf, &n));
    printf("  Vendor: %s\n", buf);

    OCL_CHECK(clGetPlatformInfo(p, CL_PLATFORM_VERSION, sizeof(buf), buf, &n));
    printf("  Version: %s\n", buf);

    OCL_CHECK(clGetPlatformInfo(p, CL_PLATFORM_PROFILE, sizeof(buf), buf, &n));
    printf("  Profile: %s\n", buf);

    OCL_CHECK(clGetPlatformInfo(p, CL_PLATFORM_EXTENSIONS, sizeof(buf), buf, &n));
    printf("  Extensions: %s\n", buf);
}

static void print_device_info(cl_device_id d) {
    char buf[8192];
    size_t n = 0;

    cl_device_type dtype;
    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL));

    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(buf), buf, &n));
    printf("  Device: %s\n", buf);
    printf("    Type: %s\n",
           (dtype & CL_DEVICE_TYPE_GPU) ? "GPU" :
           (dtype & CL_DEVICE_TYPE_CPU) ? "CPU" :
           (dtype & CL_DEVICE_TYPE_ACCELERATOR) ? "ACCELERATOR" : "OTHER");

    cl_uint cu;
    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL));
    printf("    Compute Units: %u\n", cu);

    cl_uint freq;
    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, NULL));
    printf("    Clock (MHz): %u\n", freq);

    cl_ulong gmem;
    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL));
    printf("    Global Mem: %.2f MB\n", (double)gmem / (1024.0 * 1024.0));

    cl_ulong lmem;
    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL));
    printf("    Local Mem: %.2f KB\n", (double)lmem / 1024.0);

    size_t max_wg;
    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL));
    printf("    Max Work-Group: %zu\n", max_wg);

    cl_uint dims;
    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dims), &dims, NULL));
    printf("    Max Work-Item Dims: %u\n", dims);

    size_t wis[3] = {0,0,0};
    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(wis), wis, NULL));
    printf("    Max Work-Item Sizes: %zu %zu %zu\n", wis[0], wis[1], wis[2]);

    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_VERSION, sizeof(buf), buf, &n));
    printf("    Device Version: %s\n", buf);

    OCL_CHECK(clGetDeviceInfo(d, CL_DRIVER_VERSION, sizeof(buf), buf, &n));
    printf("    Driver Version: %s\n", buf);

    OCL_CHECK(clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, &n));
    printf("    Extensions: %s\n", buf);
}

void ocl_print_device_info_like_clinfo(void) {
    cl_uint pcount = 0;
    OCL_CHECK(clGetPlatformIDs(0, NULL, &pcount));
    if (pcount == 0) {
        printf("No OpenCL platforms found.\n");
        return;
    }
    cl_platform_id* ps = (cl_platform_id*)calloc(pcount, sizeof(cl_platform_id));
    OCL_CHECK(clGetPlatformIDs(pcount, ps, NULL));

    for (cl_uint i = 0; i < pcount; ++i) {
        print_platform_info(ps[i]);

        cl_uint dcount = 0;
        cl_int err = clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_ALL, 0, NULL, &dcount);
        if (err != CL_SUCCESS || dcount == 0) {
            printf("  No devices.\n");
            continue;
        }
        cl_device_id* ds = (cl_device_id*)calloc(dcount, sizeof(cl_device_id));
        OCL_CHECK(clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_ALL, dcount, ds, NULL));

        for (cl_uint j = 0; j < dcount; ++j) {
            print_device_info(ds[j]);
        }
        free(ds);
        printf("\n");
    }
    free(ps);
}

static int pick_device(cl_platform_id* outP, cl_device_id* outD) {
    cl_uint pcount = 0;
    if (clGetPlatformIDs(0, NULL, &pcount) != CL_SUCCESS || pcount == 0) return 0;
    cl_platform_id* ps = (cl_platform_id*)calloc(pcount, sizeof(cl_platform_id));
    OCL_CHECK(clGetPlatformIDs(pcount, ps, NULL));

    // Prefer GPU
    for (cl_uint i = 0; i < pcount; ++i) {
        cl_uint dcount = 0;
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 0, NULL, &dcount) == CL_SUCCESS && dcount > 0) {
            cl_device_id d;
            OCL_CHECK(clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 1, &d, NULL));
            *outP = ps[i];
            *outD = d;
            free(ps);
            return 1;
        }
    }
    // Fallback CPU
    for (cl_uint i = 0; i < pcount; ++i) {
        cl_uint dcount = 0;
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_CPU, 0, NULL, &dcount) == CL_SUCCESS && dcount > 0) {
            cl_device_id d;
            OCL_CHECK(clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_CPU, 1, &d, NULL));
            *outP = ps[i];
            *outD = d;
            free(ps);
            return 1;
        }
    }
    free(ps);
    return 0;
}

int ocl_init_default(OclContext* ocl) {
    memset(ocl, 0, sizeof(*ocl));
    if (!pick_device(&ocl->platform, &ocl->device)) return 0;

    cl_int err = 0;
    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return 0;

#if defined(CL_VERSION_2_0)
    const cl_queue_properties props[] = {0};
    ocl->queue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, props, &err);
#else
    ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
#endif
    if (err != CL_SUCCESS) return 0;
    return 1;
}

void ocl_cleanup(OclContext* ocl) {
    if (ocl->queue) clReleaseCommandQueue(ocl->queue);
    if (ocl->context) clReleaseContext(ocl->context);
    memset(ocl, 0, sizeof(*ocl));
}

char* ocl_read_text_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (sz < 0) { fclose(f); return NULL; }
    char* data = (char*)malloc((size_t)sz + 1);
    if (!data) { fclose(f); return NULL; }
    size_t rd = fread(data, 1, (size_t)sz, f);
    fclose(f);
    data[rd] = '\0';
    if (out_size) *out_size = rd;
    return data;
}

cl_program ocl_build_program_from_file(
    const OclContext* ocl,
    const char* file_path,
    const char* build_options
) {
    size_t sz = 0;
    char* src = ocl_read_text_file(file_path, &sz);
    if (!src) {
        fprintf(stderr, "Failed to read kernel file: %s\n", file_path);
        exit(1);
    }

    cl_int err = 0;
    const char* sources[] = { src };
    cl_program prog = clCreateProgramWithSource(ocl->context, 1, sources, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource failed.\n");
        exit(1);
    }

    err = clBuildProgram(prog, 1, &ocl->device, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSz = 0;
        clGetProgramBuildInfo(prog, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSz);
        char* log = (char*)malloc(logSz + 1);
        clGetProgramBuildInfo(prog, ocl->device, CL_PROGRAM_BUILD_LOG, logSz, log, NULL);
        log[logSz] = '\0';
        fprintf(stderr, "Build failed for %s\nBuild log:\n%s\n", file_path, log);
        free(log);
        exit(1);
    }

    free(src);
    return prog;
}

cl_kernel ocl_create_kernel(cl_program program, const char* kernel_name) {
    cl_int err = 0;
    cl_kernel k = clCreateKernel(program, kernel_name, &err);
    OCL_CHECK(err);
    return k;
}

cl_mem ocl_create_buffer(cl_context ctx, cl_mem_flags flags, size_t bytes) {
    cl_int err = 0;
    cl_mem b = clCreateBuffer(ctx, flags, bytes, NULL, &err);
    OCL_CHECK(err);
    return b;
}

cl_int ocl_finish(cl_command_queue q) {
    return clFinish(q);
}