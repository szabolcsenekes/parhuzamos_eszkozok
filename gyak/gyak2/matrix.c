#include "matrix.h"
#include "profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void fill_rand(float* a, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i=0;i<n;i++) a[i] = (float)(rand()%1000)/100.0f;
}

static void seq_transpose(const float* A, float* AT, int R, int C){
    for(int r=0;r<R;r++) for(int c=0;c<C;c++) AT[c*R+r]=A[r*C+c];
}

static void seq_mul(const float* A, const float* B, float* Cc, int R, int K, int C){
    for(int r=0;r<R;r++){
        for(int c=0;c<C;c++){
            float s=0.0f;
            for(int t=0;t<K;t++) s += A[r*K+t]*B[t*C+c];
            Cc[r*C+c]=s;
        }
    }
}

static double max_abs_diff(const float* a, const float* b, size_t n){
    double m=0.0;
    for(size_t i=0;i<n;i++){
        double d=fabs((double)a[i]-(double)b[i]);
        if(d>m) m=d;
    }
    return m;
}

static void die_build_if_needed(const BuildResult* br, const char* where){
    if (br->build_err != CL_SUCCESS) {
        fprintf(stderr, "[%s] build error: %d (%s)\n", where, br->build_err, cl_error_name(br->build_err));
        if (br->build_log) fprintf(stderr, "Build log:\n%s\n", br->build_log);
        exit(1);
    }
}

void mat_transpose_demo(const Ocl* ocl, int R, int C) {
    BuildResult br = build_program_from_file(ocl, "kernels/matrix_naive.cl", "");
    die_build_if_needed(&br, "transpose");
    cl_kernel k = create_kernel_or_die(br.program, "mat_transpose");

    size_t n = (size_t)R*(size_t)C;
    float* A = (float*)malloc(sizeof(float)*n);
    float* AT = (float*)malloc(sizeof(float)*n);
    float* ref = (float*)malloc(sizeof(float)*n);
    fill_rand(A, n, 1);

    cl_int err=0;
    cl_mem aB = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*n, NULL, &err);
    cl_mem oB = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(float)*n, NULL, &err);

    cl_event evW, evK, evR;
    err = clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*n, A, 0, NULL, &evW);

    clSetKernelArg(k, 0, sizeof(aB), &aB);
    clSetKernelArg(k, 1, sizeof(oB), &oB);
    clSetKernelArg(k, 2, sizeof(int), &R);
    clSetKernelArg(k, 3, sizeof(int), &C);

    size_t g[2] = {(size_t)R, (size_t)C};
    err = clEnqueueNDRangeKernel(ocl->queue, k, 2, NULL, g, NULL, 1, &evW, &evK);

    err = clEnqueueReadBuffer(ocl->queue, oB, CL_TRUE, 0, sizeof(float)*n, AT, 1, &evK, &evR);

    seq_transpose(A, ref, R, C);
    printf("[transpose %dx%d] maxAbsDiff=%g | write=%.3fms kernel=%.3fms read=%.3fms\n",
           R, C, max_abs_diff(AT, ref, n), event_ms(evW), event_ms(evK), event_ms(evR));

    clReleaseEvent(evW); clReleaseEvent(evK); clReleaseEvent(evR);
    clReleaseMemObject(aB); clReleaseMemObject(oB);
    clReleaseKernel(k);
    free_build_result(&br);
    free(A); free(AT); free(ref);
}

void mat_mul_demo_naive(const Ocl* ocl, int R, int K, int C) {
    BuildResult br = build_program_from_file(ocl, "kernels/matrix_naive.cl", "");
    die_build_if_needed(&br, "mul_naive");
    cl_kernel kker = create_kernel_or_die(br.program, "mat_mul_naive");

    size_t nA=(size_t)R*(size_t)K, nB=(size_t)K*(size_t)C, nC=(size_t)R*(size_t)C;
    float *A=(float*)malloc(sizeof(float)*nA), *B=(float*)malloc(sizeof(float)*nB);
    float *Cc=(float*)malloc(sizeof(float)*nC), *ref=(float*)malloc(sizeof(float)*nC);
    fill_rand(A,nA,2); fill_rand(B,nB,3);

    cl_int err=0;
    cl_mem aB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*nA, NULL, &err);
    cl_mem bB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*nB, NULL, &err);
    cl_mem cB=clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,sizeof(float)*nC, NULL, &err);

    cl_event evW1, evW2, evK, evR;
    clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*nA, A, 0, NULL, &evW1);
    clEnqueueWriteBuffer(ocl->queue, bB, CL_FALSE, 0, sizeof(float)*nB, B, 0, NULL, &evW2);

    clSetKernelArg(kker,0,sizeof(aB),&aB);
    clSetKernelArg(kker,1,sizeof(bB),&bB);
    clSetKernelArg(kker,2,sizeof(cB),&cB);
    clSetKernelArg(kker,3,sizeof(int),&R);
    clSetKernelArg(kker,4,sizeof(int),&K);
    clSetKernelArg(kker,5,sizeof(int),&C);

    size_t g[2]={(size_t)R,(size_t)C};
    cl_event deps[2]={evW1,evW2};
    clEnqueueNDRangeKernel(ocl->queue, kker, 2, NULL, g, NULL, 2, deps, &evK);
    clEnqueueReadBuffer(ocl->queue, cB, CL_TRUE, 0, sizeof(float)*nC, Cc, 1, &evK, &evR);

    seq_mul(A,B,ref,R,K,C);
    printf("[mul_naive %dx%dx%d] maxAbsDiff=%g | write=%.3fms kernel=%.3fms read=%.3fms\n",
           R,K,C, max_abs_diff(Cc,ref,nC),
           event_ms(evW1)+event_ms(evW2), event_ms(evK), event_ms(evR));

    clReleaseEvent(evW1); clReleaseEvent(evW2); clReleaseEvent(evK); clReleaseEvent(evR);
    clReleaseMemObject(aB); clReleaseMemObject(bB); clReleaseMemObject(cB);
    clReleaseKernel(kker);
    free_build_result(&br);
    free(A); free(B); free(Cc); free(ref);
}

void mat_mul_demo_tiled(const Ocl* ocl, int R, int K, int C, int tile) {
    char opts[64];
    snprintf(opts, sizeof(opts), "-DTILE=%d", tile);

    BuildResult br = build_program_from_file(ocl, "kernels/matrix_tiled.cl", opts);
    die_build_if_needed(&br, "mul_tiled");
    cl_kernel kker = create_kernel_or_die(br.program, "mat_mul_tiled");

    size_t nA=(size_t)R*(size_t)K, nB=(size_t)K*(size_t)C, nC=(size_t)R*(size_t)C;
    float *A=(float*)malloc(sizeof(float)*nA), *B=(float*)malloc(sizeof(float)*nB);
    float *Cc=(float*)malloc(sizeof(float)*nC), *ref=(float*)malloc(sizeof(float)*nC);
    fill_rand(A,nA,4); fill_rand(B,nB,5);

    cl_int err=0;
    cl_mem aB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*nA, NULL, &err);
    cl_mem bB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*nB, NULL, &err);
    cl_mem cB=clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,sizeof(float)*nC, NULL, &err);

    cl_event evW1, evW2, evK, evR;
    clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*nA, A, 0, NULL, &evW1);
    clEnqueueWriteBuffer(ocl->queue, bB, CL_FALSE, 0, sizeof(float)*nB, B, 0, NULL, &evW2);

    clSetKernelArg(kker,0,sizeof(aB),&aB);
    clSetKernelArg(kker,1,sizeof(bB),&bB);
    clSetKernelArg(kker,2,sizeof(cB),&cB);
    clSetKernelArg(kker,3,sizeof(int),&R);
    clSetKernelArg(kker,4,sizeof(int),&K);
    clSetKernelArg(kker,5,sizeof(int),&C);

    size_t local[2]={(size_t)tile,(size_t)tile};
    size_t global[2]={
        (size_t)((R + tile - 1) / tile) * (size_t)tile,
        (size_t)((C + tile - 1) / tile) * (size_t)tile
    };

    cl_event deps[2]={evW1,evW2};
    clEnqueueNDRangeKernel(ocl->queue, kker, 2, NULL, global, local, 2, deps, &evK);
    clEnqueueReadBuffer(ocl->queue, cB, CL_TRUE, 0, sizeof(float)*nC, Cc, 1, &evK, &evR);

    // correctness vs seq (kisebb méreteknél ajánlott; nagy N-nél lassú)
    seq_mul(A,B,ref,R,K,C);
    printf("[mul_tiled T=%d %dx%dx%d] maxAbsDiff=%g | write=%.3fms kernel=%.3fms read=%.3fms\n",
           tile,R,K,C, max_abs_diff(Cc,ref,nC),
           event_ms(evW1)+event_ms(evW2), event_ms(evK), event_ms(evR));

    clReleaseEvent(evW1); clReleaseEvent(evW2); clReleaseEvent(evK); clReleaseEvent(evR);
    clReleaseMemObject(aB); clReleaseMemObject(bB); clReleaseMemObject(cB);
    clReleaseKernel(kker);
    free_build_result(&br);
    free(A); free(B); free(Cc); free(ref);
}

void mat_row_sum_demo(const Ocl* ocl, int R, int C) {
    BuildResult br = build_program_from_file(ocl, "kernels/matrix_naive.cl", "");
    die_build_if_needed(&br, "row_sum");
    cl_kernel k = create_kernel_or_die(br.program, "row_sum");

    size_t n=(size_t)R*(size_t)C;
    float* A=(float*)malloc(sizeof(float)*n);
    float* out=(float*)malloc(sizeof(float)*(size_t)R);
    fill_rand(A,n,6);

    cl_int err=0;
    cl_mem aB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*n, NULL, &err);
    cl_mem oB=clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,sizeof(float)*(size_t)R, NULL, &err);

    cl_event evW, evK, evR;
    clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*n, A, 0, NULL, &evW);

    clSetKernelArg(k,0,sizeof(aB),&aB);
    clSetKernelArg(k,1,sizeof(oB),&oB);
    clSetKernelArg(k,2,sizeof(int),&R);
    clSetKernelArg(k,3,sizeof(int),&C);

    size_t g=(size_t)R;
    clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 1, &evW, &evK);
    clEnqueueReadBuffer(ocl->queue, oB, CL_TRUE, 0, sizeof(float)*(size_t)R, out, 1, &evK, &evR);

    printf("[row_sum %dx%d] write=%.3fms kernel=%.3fms read=%.3fms | out[0]=%f\n",
           R,C,event_ms(evW),event_ms(evK),event_ms(evR),out[0]);

    clReleaseEvent(evW); clReleaseEvent(evK); clReleaseEvent(evR);
    clReleaseMemObject(aB); clReleaseMemObject(oB);
    clReleaseKernel(k);
    free_build_result(&br);
    free(A); free(out);
}

void mat_col_sum_demo(const Ocl* ocl, int R, int C) {
    BuildResult br = build_program_from_file(ocl, "kernels/matrix_naive.cl", "");
    die_build_if_needed(&br, "col_sum");
    cl_kernel k = create_kernel_or_die(br.program, "col_sum");

    size_t n=(size_t)R*(size_t)C;
    float* A=(float*)malloc(sizeof(float)*n);
    float* out=(float*)malloc(sizeof(float)*(size_t)C);
    fill_rand(A,n,7);

    cl_int err=0;
    cl_mem aB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*n, NULL, &err);
    cl_mem oB=clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,sizeof(float)*(size_t)C, NULL, &err);

    cl_event evW, evK, evR;
    clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*n, A, 0, NULL, &evW);

    clSetKernelArg(k,0,sizeof(aB),&aB);
    clSetKernelArg(k,1,sizeof(oB),&oB);
    clSetKernelArg(k,2,sizeof(int),&R);
    clSetKernelArg(k,3,sizeof(int),&C);

    size_t g=(size_t)C;
    clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 1, &evW, &evK);
    clEnqueueReadBuffer(ocl->queue, oB, CL_TRUE, 0, sizeof(float)*(size_t)C, out, 1, &evK, &evR);

    printf("[col_sum %dx%d] write=%.3fms kernel=%.3fms read=%.3fms | out[0]=%f\n",
           R,C,event_ms(evW),event_ms(evK),event_ms(evR),out[0]);

    clReleaseEvent(evW); clReleaseEvent(evK); clReleaseEvent(evR);
    clReleaseMemObject(aB); clReleaseMemObject(oB);
    clReleaseKernel(k);
    free_build_result(&br);
    free(A); free(out);
}

// 5) benchmark + CSV: komplexitás + adatmozgatás
void benchmark_matrix(const Ocl* ocl, int Nmin, int Nmax, int step) {
    FILE* f = fopen("bench_matrix.csv", "wb");
    if (!f) { printf("Cannot open bench_matrix.csv\n"); return; }
    fprintf(f, "N,variant,tile,write_ms,kernel_ms,read_ms,total_ms\n");

    for (int N=Nmin; N<=Nmax; N+=step) {
        // Naiv
        {
            BuildResult br = build_program_from_file(ocl, "kernels/matrix_naive.cl", "");
            die_build_if_needed(&br, "bench_naive");
            cl_kernel k = create_kernel_or_die(br.program, "mat_mul_naive");

            size_t nA=(size_t)N*(size_t)N, nB=nA, nC=nA;
            float *A=(float*)malloc(sizeof(float)*nA), *B=(float*)malloc(sizeof(float)*nB), *C=(float*)malloc(sizeof(float)*nC);
            fill_rand(A,nA,10); fill_rand(B,nB,11);

            cl_int err=0;
            cl_mem aB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*nA, NULL, &err);
            cl_mem bB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*nB, NULL, &err);
            cl_mem cB=clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,sizeof(float)*nC, NULL, &err);

            cl_event evW1, evW2, evK, evR;
            clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*nA, A, 0, NULL, &evW1);
            clEnqueueWriteBuffer(ocl->queue, bB, CL_FALSE, 0, sizeof(float)*nB, B, 0, NULL, &evW2);

            clSetKernelArg(k,0,sizeof(aB),&aB);
            clSetKernelArg(k,1,sizeof(bB),&bB);
            clSetKernelArg(k,2,sizeof(cB),&cB);
            clSetKernelArg(k,3,sizeof(int),&N);
            clSetKernelArg(k,4,sizeof(int),&N);
            clSetKernelArg(k,5,sizeof(int),&N);

            size_t g[2]={(size_t)N,(size_t)N};
            cl_event deps[2]={evW1,evW2};
            clEnqueueNDRangeKernel(ocl->queue, k, 2, NULL, g, NULL, 2, deps, &evK);
            clEnqueueReadBuffer(ocl->queue, cB, CL_TRUE, 0, sizeof(float)*nC, C, 1, &evK, &evR);

            double w = event_ms(evW1)+event_ms(evW2);
            double km = event_ms(evK);
            double r = event_ms(evR);
            fprintf(f, "%d,naive,0,%.6f,%.6f,%.6f,%.6f\n", N, w, km, r, w+km+r);

            clReleaseEvent(evW1); clReleaseEvent(evW2); clReleaseEvent(evK); clReleaseEvent(evR);
            clReleaseMemObject(aB); clReleaseMemObject(bB); clReleaseMemObject(cB);
            clReleaseKernel(k);
            free_build_result(&br);
            free(A); free(B); free(C);
        }

        // Tiled (tile=16)
        for (int tile=8; tile<=16; tile*=2) {
            char opts[64];
            snprintf(opts, sizeof(opts), "-DTILE=%d", tile);
            BuildResult br = build_program_from_file(ocl, "kernels/matrix_tiled.cl", opts);
            die_build_if_needed(&br, "bench_tiled");
            cl_kernel k = create_kernel_or_die(br.program, "mat_mul_tiled");

            size_t nA=(size_t)N*(size_t)N, nB=nA, nC=nA;
            float *A=(float*)malloc(sizeof(float)*nA), *B=(float*)malloc(sizeof(float)*nB), *C=(float*)malloc(sizeof(float)*nC);
            fill_rand(A,nA,20+tile); fill_rand(B,nB,30+tile);

            cl_int err=0;
            cl_mem aB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*nA, NULL, &err);
            cl_mem bB=clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*nB, NULL, &err);
            cl_mem cB=clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,sizeof(float)*nC, NULL, &err);

            cl_event evW1, evW2, evK, evR;
            clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*nA, A, 0, NULL, &evW1);
            clEnqueueWriteBuffer(ocl->queue, bB, CL_FALSE, 0, sizeof(float)*nB, B, 0, NULL, &evW2);

            clSetKernelArg(k,0,sizeof(aB),&aB);
            clSetKernelArg(k,1,sizeof(bB),&bB);
            clSetKernelArg(k,2,sizeof(cB),&cB);
            clSetKernelArg(k,3,sizeof(int),&N);
            clSetKernelArg(k,4,sizeof(int),&N);
            clSetKernelArg(k,5,sizeof(int),&N);

            size_t local[2]={(size_t)tile,(size_t)tile};
            size_t global[2]={
                (size_t)((N + tile - 1)/tile)*tile,
                (size_t)((N + tile - 1)/tile)*tile
            };

            cl_event deps[2]={evW1,evW2};
            clEnqueueNDRangeKernel(ocl->queue, k, 2, NULL, global, local, 2, deps, &evK);
            clEnqueueReadBuffer(ocl->queue, cB, CL_TRUE, 0, sizeof(float)*nC, C, 1, &evK, &evR);

            double w = event_ms(evW1)+event_ms(evW2);
            double km = event_ms(evK);
            double r = event_ms(evR);
            fprintf(f, "%d,tiled,%d,%.6f,%.6f,%.6f,%.6f\n", N, tile, w, km, r, w+km+r);

            clReleaseEvent(evW1); clReleaseEvent(evW2); clReleaseEvent(evK); clReleaseEvent(evR);
            clReleaseMemObject(aB); clReleaseMemObject(bB); clReleaseMemObject(cB);
            clReleaseKernel(k);
            free_build_result(&br);
            free(A); free(B); free(C);
        }

        fflush(f);
        printf("[bench] N=%d done\n", N);
    }

    fclose(f);
    printf("Wrote bench_matrix.csv (plot in Excel/LibreOffice)\n");
}