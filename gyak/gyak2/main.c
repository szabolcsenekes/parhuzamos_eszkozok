#include "ocl.h"
#include "profile.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* exe) {
    printf("Usage:\n");
    printf("  %s errcodes\n", exe);
    printf("  %s error_demo huge_kernel\n", exe);
    printf("  %s error_demo div0_float\n", exe);
    printf("  %s error_demo div0_int\n", exe);
    printf("  %s error_demo bad_recursion\n", exe);
    printf("  %s hello_profile N\n", exe);
    printf("  %s mat transpose R C\n", exe);
    printf("  %s mat mul_naive R K C\n", exe);
    printf("  %s mat mul_tiled R K C TILE\n", exe);
    printf("  %s mat row_sum R C\n", exe);
    printf("  %s mat col_sum R C\n", exe);
    printf("  %s bench Nmin Nmax step\n", exe);
}

static char* make_huge_kernel_source(size_t bytes) {
    char* s = (char*)malloc(bytes + 1);
    if (!s) return NULL;
    size_t pos = 0;
    pos += (size_t)snprintf(s+pos, bytes-pos,
        "__kernel void k(__global int* a){ int i=(int)get_global_id(0); a[i]=i; }\n");
    while (pos + 64 < bytes) {
        pos += (size_t)snprintf(s+pos, bytes-pos, "/* padding padding padding padding padding */\n");
    }
    s[pos] = 0;
    return s;
}

static void demo_error_huge_kernel(const Ocl* ocl) {
    // próbálj 50-200MB között; driverfüggő, mikor dől el
    size_t bytes = 100u * 1024u * 1024u;
    char* src = make_huge_kernel_source(bytes);
    if (!src) { printf("malloc failed for huge kernel\n"); return; }

    cl_int err=0;
    const char* arr[1]={src};
    cl_program p = clCreateProgramWithSource(ocl->context, 1, arr, NULL, &err);
    printf("[huge_kernel] clCreateProgramWithSource: %d (%s)\n", err, cl_error_name(err));
    if (err == CL_SUCCESS) {
        err = clBuildProgram(p, 1, &ocl->device, "", NULL, NULL);
        printf("[huge_kernel] clBuildProgram: %d (%s)\n", err, cl_error_name(err));
        // build log
        size_t sz=0;
        clGetProgramBuildInfo(p, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
        if (sz > 1) {
            char* log=(char*)malloc(sz+1);
            clGetProgramBuildInfo(p, ocl->device, CL_PROGRAM_BUILD_LOG, sz, log, NULL);
            log[sz]=0;
            printf("Build log (first 2000 chars max):\n");
            for (size_t i=0;i<sz && i<2000;i++) putchar(log[i]);
            putchar('\n');
            free(log);
        }
        clReleaseProgram(p);
    }
    free(src);
}

static void demo_error_from_file(const Ocl* ocl, const char* kernelName) {
    BuildResult br = build_program_from_file(ocl, "kernels/errors.cl", "");
    printf("[error_demo %s] build: %d (%s)\n", kernelName, br.build_err, cl_error_name(br.build_err));
    if (br.build_log) printf("Build log:\n%s\n", br.build_log);

    if (br.build_err != CL_SUCCESS) { free_build_result(&br); return; }

    cl_kernel k = create_kernel_or_die(br.program, kernelName);

    // futtatás kis N-en
    const int N=8;
    cl_int err=0;

    if (strcmp(kernelName, "div0_float")==0) {
        cl_mem outB = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(float)*N, NULL, &err);
        clSetKernelArg(k, 0, sizeof(outB), &outB);
        size_t g=(size_t)N;
        err = clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL);
        printf("enqueue: %d (%s)\n", err, cl_error_name(err));
        float out[N];
        clEnqueueReadBuffer(ocl->queue, outB, CL_TRUE, 0, sizeof(out), out, 0, NULL, NULL);
        printf("result: ");
        for(int i=0;i<N;i++) printf("%f ", out[i]);
        printf("\n");
        clReleaseMemObject(outB);
    } else if (strcmp(kernelName, "div0_int")==0) {
        cl_mem outB = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(int)*N, NULL, &err);
        clSetKernelArg(k, 0, sizeof(outB), &outB);
        size_t g=(size_t)N;
        err = clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL);
        printf("enqueue: %d (%s)\n", err, cl_error_name(err));
        int out[N];
        clEnqueueReadBuffer(ocl->queue, outB, CL_TRUE, 0, sizeof(out), out, 0, NULL, NULL);
        printf("result: ");
        for(int i=0;i<N;i++) printf("%d ", out[i]);
        printf("\n");
        clReleaseMemObject(outB);
    } else {
        printf("Kernel %s not executed (demo focuses on build/runtime errors).\n", kernelName);
    }

    clReleaseKernel(k);
    free_build_result(&br);
}

static void hello_profile(const Ocl* ocl, int N) {
    BuildResult br = build_program_from_file(ocl, "kernels/hello_add.cl", "");
    if (br.build_err != CL_SUCCESS) {
        printf("Build failed: %d (%s)\n", br.build_err, cl_error_name(br.build_err));
        if (br.build_log) printf("Build log:\n%s\n", br.build_log);
        free_build_result(&br);
        return;
    }
    cl_kernel k = create_kernel_or_die(br.program, "hello_add");

    float* a=(float*)malloc(sizeof(float)*(size_t)N);
    float* b=(float*)malloc(sizeof(float)*(size_t)N);
    float* o=(float*)malloc(sizeof(float)*(size_t)N);
    for(int i=0;i<N;i++){ a[i]= (float)(i%100)/10.0f; b[i]=(float)((i+1)%100)/10.0f; }

    cl_int err=0;
    cl_mem aB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*(size_t)N, NULL, &err);
    cl_mem bB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*(size_t)N, NULL, &err);
    cl_mem oB=clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,sizeof(float)*(size_t)N, NULL, &err);

    cl_event evW1, evW2, evK, evR;
    clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*(size_t)N, a, 0, NULL, &evW1);
    clEnqueueWriteBuffer(ocl->queue, bB, CL_FALSE, 0, sizeof(float)*(size_t)N, b, 0, NULL, &evW2);

    clSetKernelArg(k,0,sizeof(aB),&aB);
    clSetKernelArg(k,1,sizeof(bB),&bB);
    clSetKernelArg(k,2,sizeof(oB),&oB);
    clSetKernelArg(k,3,sizeof(int),&N);

    size_t g=(size_t)N;
    cl_event deps[2]={evW1,evW2};
    clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 2, deps, &evK);
    clEnqueueReadBuffer(ocl->queue, oB, CL_TRUE, 0, sizeof(float)*(size_t)N, o, 1, &evK, &evR);

    double t_write = event_ms(evW1)+event_ms(evW2);
    double t_kernel= event_ms(evK);
    double t_read  = event_ms(evR);

    printf("[hello_profile] N=%d write=%.3fms kernel=%.3fms read=%.3fms total=%.3fms\n",
           N,t_write,t_kernel,t_read,t_write+t_kernel+t_read);

    write_csv_hello_profile("hello_profile.csv", t_write, t_kernel, t_read);
    printf("Wrote hello_profile.csv (plot: step vs ms)\n");

    clReleaseEvent(evW1); clReleaseEvent(evW2); clReleaseEvent(evK); clReleaseEvent(evR);
    clReleaseMemObject(aB); clReleaseMemObject(bB); clReleaseMemObject(oB);
    clReleaseKernel(k);
    free_build_result(&br);
    free(a); free(b); free(o);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 0; }

    if (strcmp(argv[1], "errcodes")==0) {
        printf("Example error name: %s\n", cl_error_name(CL_INVALID_VALUE));
        printf("OpenCL errors are defined in CL/cl.h and related headers.\n");
        return 0;
    }

    Ocl ocl;
    if (!ocl_init(&ocl)) {
        printf("OpenCL init failed (no platform/device?)\n");
        return 1;
    }

    if (strcmp(argv[1], "error_demo")==0) {
        if (argc < 3) { usage(argv[0]); ocl_cleanup(&ocl); return 0; }
        if (strcmp(argv[2], "huge_kernel")==0) demo_error_huge_kernel(&ocl);
        else if (strcmp(argv[2], "div0_float")==0) demo_error_from_file(&ocl, "div0_float");
        else if (strcmp(argv[2], "div0_int")==0) demo_error_from_file(&ocl, "div0_int");
        else if (strcmp(argv[2], "bad_recursion")==0) demo_error_from_file(&ocl, "bad_recursion");
        else usage(argv[0]);
    }
    else if (strcmp(argv[1], "hello_profile")==0) {
        int N = (argc>=3) ? atoi(argv[2]) : 1000000;
        hello_profile(&ocl, N);
    }
    else if (strcmp(argv[1], "mat")==0) {
        if (argc < 3) { usage(argv[0]); ocl_cleanup(&ocl); return 0; }
        if (strcmp(argv[2], "transpose")==0 && argc>=5) mat_transpose_demo(&ocl, atoi(argv[3]), atoi(argv[4]));
        else if (strcmp(argv[2], "mul_naive")==0 && argc>=6) mat_mul_demo_naive(&ocl, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        else if (strcmp(argv[2], "mul_tiled")==0 && argc>=7) mat_mul_demo_tiled(&ocl, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
        else if (strcmp(argv[2], "row_sum")==0 && argc>=5) mat_row_sum_demo(&ocl, atoi(argv[3]), atoi(argv[4]));
        else if (strcmp(argv[2], "col_sum")==0 && argc>=5) mat_col_sum_demo(&ocl, atoi(argv[3]), atoi(argv[4]));
        else usage(argv[0]);
    }
    else if (strcmp(argv[1], "bench")==0) {
        int Nmin = (argc>=3) ? atoi(argv[2]) : 256;
        int Nmax = (argc>=4) ? atoi(argv[3]) : 1024;
        int step = (argc>=5) ? atoi(argv[4]) : 256;
        benchmark_matrix(&ocl, Nmin, Nmax, step);
    }
    else {
        usage(argv[0]);
    }

    ocl_cleanup(&ocl);
    return 0;
}