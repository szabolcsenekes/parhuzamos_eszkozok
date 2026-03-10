#include "task.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OCL_CHECK(e) do { \
    cl_int _err = (e); \
    if (_err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d (%s) at %s:%d\n", _err, ocl_errstr(_err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static void fill_random_f(float* a, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i=0;i<n;i++) a[i] = (float)(rand()%10000) / 100.0f;
}

static void fill_random_u32(uint32_t* a, size_t n, unsigned seed, uint32_t mod) {
    srand(seed);
    for (size_t i=0;i<n;i++) a[i] = (uint32_t)(rand() % (int)mod);
}

/* ========================= 3) Mapping ========================= */

void task_map_index(const OclContext* ocl, int use_local_index, int n) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/map_index.cl", "");
    cl_kernel k = ocl_create_kernel(p, use_local_index ? "map_local_index" : "map_global_index");

    cl_mem out = ocl_create_buffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(int)* (size_t)n);

    OCL_CHECK(clSetKernelArg(k, 0, sizeof(out), &out));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(int), &n));

    size_t g = (size_t)n;
    size_t l = 0; // runtime dönti el
    OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL));

    int* h = (int*)malloc(sizeof(int)*(size_t)n);
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, out, CL_TRUE, 0, sizeof(int)*(size_t)n, h, 0, NULL, NULL));

    printf("[mapping index] n=%d (%s)\n", n, use_local_index ? "local" : "global");
    for (int i=0;i< (n<16?n:16); i++) printf("  out[%d]=%d\n", i, h[i]);

    free(h);
    clReleaseMemObject(out);
    clReleaseKernel(k);
    clReleaseProgram(p);
}

void task_map_reverse(const OclContext* ocl, int n) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/map_reverse.cl", "");
    cl_kernel k = ocl_create_kernel(p, "map_reverse");

    int* inH = (int*)malloc(sizeof(int)*(size_t)n);
    for (int i=0;i<n;i++) inH[i]=i;

    cl_mem inB  = ocl_create_buffer(ocl->context, CL_MEM_READ_ONLY, sizeof(int)*(size_t)n);
    cl_mem outB = ocl_create_buffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(int)*(size_t)n);

    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, inB, CL_TRUE, 0, sizeof(int)*(size_t)n, inH, 0, NULL, NULL));

    OCL_CHECK(clSetKernelArg(k, 0, sizeof(inB), &inB));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(outB), &outB));
    OCL_CHECK(clSetKernelArg(k, 2, sizeof(int), &n));

    size_t g=(size_t)n;
    OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL));

    int* outH = (int*)malloc(sizeof(int)*(size_t)n);
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, outB, CL_TRUE, 0, sizeof(int)*(size_t)n, outH, 0, NULL, NULL));

    printf("[mapping reverse] in[0..7]=0..7, out[0..7]=\n");
    for (int i=0;i< (n<8?n:8); i++) printf("  out[%d]=%d\n", i, outH[i]);

    free(inH); free(outH);
    clReleaseMemObject(inB);
    clReleaseMemObject(outB);
    clReleaseKernel(k);
    clReleaseProgram(p);
}

void task_map_swap_neighbors(const OclContext* ocl, int n) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/map_swap_neighbors.cl", "");
    cl_kernel k = ocl_create_kernel(p, "swap_neighbors");

    int* inH=(int*)malloc(sizeof(int)*(size_t)n);
    for(int i=0;i<n;i++) inH[i]=i;

    cl_mem inB  = ocl_create_buffer(ocl->context, CL_MEM_READ_ONLY, sizeof(int)*(size_t)n);
    cl_mem outB = ocl_create_buffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(int)*(size_t)n);
    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, inB, CL_TRUE, 0, sizeof(int)*(size_t)n, inH, 0, NULL, NULL));

    OCL_CHECK(clSetKernelArg(k, 0, sizeof(inB), &inB));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(outB), &outB));
    OCL_CHECK(clSetKernelArg(k, 2, sizeof(int), &n));

    size_t g=(size_t)n;
    OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL));

    int* outH=(int*)malloc(sizeof(int)*(size_t)n);
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, outB, CL_TRUE, 0, sizeof(int)*(size_t)n, outH, 0, NULL, NULL));

    printf("[swap neighbors] first 10:\n");
    for (int i=0;i< (n<10?n:10); i++) printf("  %d -> %d\n", inH[i], outH[i]);

    free(inH); free(outH);
    clReleaseMemObject(inB);
    clReleaseMemObject(outB);
    clReleaseKernel(k);
    clReleaseProgram(p);
}

/* ========================= 4) Vektorösszeadás ========================= */
/*
  A trükk: vec_add(out,a,b,n) "normál" függvény, belül OpenCL-t használ.
  (Egyszerűség kedvéért minden hívásnál init+build; ha gyorsítani akarod,
   csinálhatunk perzisztens context+program cache-t.)
*/

int vec_add(float* out, const float* a, const float* b, size_t n) {
    OclContext ocl;
    if (!ocl_init_default(&ocl)) return 0;

    cl_program p = ocl_build_program_from_file(&ocl, "kernels/vec_add.cl", "");
    cl_kernel k = ocl_create_kernel(p, "vec_add");

    cl_mem aB = ocl_create_buffer(ocl.context, CL_MEM_READ_ONLY,  sizeof(float)*n);
    cl_mem bB = ocl_create_buffer(ocl.context, CL_MEM_READ_ONLY,  sizeof(float)*n);
    cl_mem oB = ocl_create_buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*n);

    OCL_CHECK(clEnqueueWriteBuffer(ocl.queue, aB, CL_TRUE, 0, sizeof(float)*n, a, 0, NULL, NULL));
    OCL_CHECK(clEnqueueWriteBuffer(ocl.queue, bB, CL_TRUE, 0, sizeof(float)*n, b, 0, NULL, NULL));

    int ni = (int)n;
    OCL_CHECK(clSetKernelArg(k, 0, sizeof(aB), &aB));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(bB), &bB));
    OCL_CHECK(clSetKernelArg(k, 2, sizeof(oB), &oB));
    OCL_CHECK(clSetKernelArg(k, 3, sizeof(int), &ni));

    size_t g = n;
    OCL_CHECK(clEnqueueNDRangeKernel(ocl.queue, k, 1, NULL, &g, NULL, 0, NULL, NULL));
    OCL_CHECK(clEnqueueReadBuffer(ocl.queue, oB, CL_TRUE, 0, sizeof(float)*n, out, 0, NULL, NULL));

    clReleaseMemObject(aB); clReleaseMemObject(bB); clReleaseMemObject(oB);
    clReleaseKernel(k); clReleaseProgram(p);
    ocl_cleanup(&ocl);
    return 1;
}

static void vec_add_seq(float* out, const float* a, const float* b, size_t n) {
    for (size_t i=0;i<n;i++) out[i]=a[i]+b[i];
}

void task_vec_add_demo(const OclContext* ocl, int n) {
    (void)ocl; // demo a "rejtett" API-t mutatja
    float* a=(float*)malloc(sizeof(float)*(size_t)n);
    float* b=(float*)malloc(sizeof(float)*(size_t)n);
    float* o=(float*)malloc(sizeof(float)*(size_t)n);
    float* ref=(float*)malloc(sizeof(float)*(size_t)n);

    fill_random_f(a, (size_t)n, 1);
    fill_random_f(b, (size_t)n, 2);

    if (!vec_add(o, a, b, (size_t)n)) {
        printf("vec_add failed (no OpenCL?).\n");
        goto done;
    }
    vec_add_seq(ref, a, b, (size_t)n);

    double maxAbs=0.0;
    for(int i=0;i<n;i++){
        double d=fabs((double)o[i]-(double)ref[i]);
        if(d>maxAbs) maxAbs=d;
    }
    printf("[vec_add] n=%d, maxAbsError=%g\n", n, maxAbs);
    for (int i=0;i< (n<8?n:8); i++) printf("  %f + %f = %f\n", a[i], b[i], o[i]);

done:
    free(a); free(b); free(o); free(ref);
}

/* ========================= 5) Hiányzó elemek pótlása ========================= */

void make_missing_input(uint32_t* a, uint8_t* missingMask, size_t n, unsigned seed, int holes) {
    // feltétel: ha hiányzik i, akkor i-1 és i+1 nem hiányozhat
    memset(missingMask, 0, n);
    fill_random_u32(a, n, seed, 1000);

    srand(seed + 123);
    int placed = 0;
    while (placed < holes) {
        size_t i = (size_t)(1 + rand() % (int)(n-2));
        if (missingMask[i]) continue;
        if (missingMask[i-1] || missingMask[i+1]) continue;
        missingMask[i] = 1;
        a[i] = 0; // "lyuk"
        placed++;
    }
}

void task_fill_missing(const OclContext* ocl, int n) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/fill_missing.cl", "");
    cl_kernel k = ocl_create_kernel(p, "fill_missing");

    uint32_t* a=(uint32_t*)malloc(sizeof(uint32_t)*(size_t)n);
    uint8_t* miss=(uint8_t*)malloc(sizeof(uint8_t)*(size_t)n);

    make_missing_input(a, miss, (size_t)n, 7, n/10);

    cl_mem aB = ocl_create_buffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_uint)*(size_t)n);
    cl_mem mB = ocl_create_buffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(cl_uchar)*(size_t)n);

    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, aB, CL_TRUE, 0, sizeof(cl_uint)*(size_t)n, a, 0, NULL, NULL));
    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, mB, CL_TRUE, 0, sizeof(cl_uchar)*(size_t)n, miss, 0, NULL, NULL));

    OCL_CHECK(clSetKernelArg(k, 0, sizeof(aB), &aB));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(mB), &mB));
    OCL_CHECK(clSetKernelArg(k, 2, sizeof(int), &n));

    size_t g=(size_t)n;
    OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL));
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, aB, CL_TRUE, 0, sizeof(cl_uint)*(size_t)n, a, 0, NULL, NULL));

    printf("[fill_missing] show first 20 (x=filled positions originally missing):\n");
    int shown=0;
    for (int i=0;i<n && shown<20;i++){
        if (miss[i]) printf("  i=%d  filled=%u (avg of %u and %u)\n", i, a[i], a[i-1], a[i+1]), shown++;
    }
    if (shown==0) printf("  (no holes?)\n");

    free(a); free(miss);
    clReleaseMemObject(aB); clReleaseMemObject(mB);
    clReleaseKernel(k); clReleaseProgram(p);
}

/* ========================= 6) Rang ========================= */

void task_rank(const OclContext* ocl, int n) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/rank.cl", "");
    cl_kernel k = ocl_create_kernel(p, "rank_all");

    int* a=(int*)malloc(sizeof(int)*(size_t)n);
    int* r=(int*)malloc(sizeof(int)*(size_t)n);
    srand(3);
    for(int i=0;i<n;i++) a[i]=rand()%100;

    cl_mem aB = ocl_create_buffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(cl_int)*(size_t)n);
    cl_mem rB = ocl_create_buffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*(size_t)n);

    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, aB, CL_TRUE, 0, sizeof(cl_int)*(size_t)n, a, 0, NULL, NULL));

    OCL_CHECK(clSetKernelArg(k, 0, sizeof(aB), &aB));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(rB), &rB));
    OCL_CHECK(clSetKernelArg(k, 2, sizeof(int), &n));

    size_t g=(size_t)n;
    OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL));
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, rB, CL_TRUE, 0, sizeof(cl_int)*(size_t)n, r, 0, NULL, NULL));

    printf("[rank] first 10:\n");
    for (int i=0;i< (n<10?n:10); i++) printf("  a[%d]=%d rank=%d\n", i, a[i], r[i]);

    free(a); free(r);
    clReleaseMemObject(aB); clReleaseMemObject(rB);
    clReleaseKernel(k); clReleaseProgram(p);
}

/* ========================= 7) Előfordulások + egyediség ========================= */

void task_occurrence(const OclContext* ocl, int n) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/occurrence.cl", "");
    cl_kernel k = ocl_create_kernel(p, "count_occurrence");

    int* a=(int*)malloc(sizeof(int)*(size_t)n);
    int* c=(int*)malloc(sizeof(int)*(size_t)n);
    srand(4);
    for(int i=0;i<n;i++) a[i]=rand()%20; // sok ütközés

    cl_mem aB = ocl_create_buffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(cl_int)*(size_t)n);
    cl_mem cB = ocl_create_buffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*(size_t)n);
    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, aB, CL_TRUE, 0, sizeof(cl_int)*(size_t)n, a, 0, NULL, NULL));

    OCL_CHECK(clSetKernelArg(k, 0, sizeof(aB), &aB));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(cB), &cB));
    OCL_CHECK(clSetKernelArg(k, 2, sizeof(int), &n));

    size_t g=(size_t)n;
    OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL));
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, cB, CL_TRUE, 0, sizeof(cl_int)*(size_t)n, c, 0, NULL, NULL));

    int allUnique = 1;
    for(int i=0;i<n;i++) if (c[i] != 1) { allUnique=0; break; }

    printf("[occurrence] allUnique=%s, first 12:\n", allUnique ? "YES" : "NO");
    for (int i=0;i< (n<12?n:12); i++) printf("  a[%d]=%d count=%d\n", i, a[i], c[i]);

    free(a); free(c);
    clReleaseMemObject(aB); clReleaseMemObject(cB);
    clReleaseKernel(k); clReleaseProgram(p);
}

/* ========================= 8) Min/Max redukció ========================= */

void task_minmax(const OclContext* ocl, int n) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/minmax_reduce.cl", "");
    cl_kernel k = ocl_create_kernel(p, "minmax_reduce_pass");

    int* a=(int*)malloc(sizeof(int)*(size_t)n);
    srand(5);
    for(int i=0;i<n;i++) a[i]= (rand()%20000) - 10000;

    // 1st pass: per work-group output
    cl_mem inB = ocl_create_buffer(ocl->context, CL_MEM_READ_ONLY, sizeof(cl_int)*(size_t)n);
    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, inB, CL_TRUE, 0, sizeof(cl_int)*(size_t)n, a, 0, NULL, NULL));

    size_t local = 256;
    size_t groups = ((size_t)n + local - 1) / local;
    cl_mem outMinB = ocl_create_buffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*groups);
    cl_mem outMaxB = ocl_create_buffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*groups);

    OCL_CHECK(clSetKernelArg(k, 0, sizeof(inB), &inB));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(outMinB), &outMinB));
    OCL_CHECK(clSetKernelArg(k, 2, sizeof(outMaxB), &outMaxB));
    OCL_CHECK(clSetKernelArg(k, 3, sizeof(int), &n));
    OCL_CHECK(clSetKernelArg(k, 4, sizeof(cl_int)*local, NULL)); // local min
    OCL_CHECK(clSetKernelArg(k, 5, sizeof(cl_int)*local, NULL)); // local max

    size_t global = groups * local;
    OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &global, &local, 0, NULL, NULL));
    
    // read partials and finalize on CPU (1 core) to minimize CPU core usage
    int* mins=(int*)malloc(sizeof(int)*groups);
    int* maxs=(int*)malloc(sizeof(int)*groups);
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, outMinB, CL_TRUE, 0, sizeof(int)*groups, mins, 0, NULL, NULL));
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, outMaxB, CL_TRUE, 0, sizeof(int)*groups, maxs, 0, NULL, NULL));

    int mn=mins[0], mx=maxs[0];
    for(size_t i=1;i<groups;i++){ if(mins[i]<mn) mn=mins[i]; if(maxs[i]>mx) mx=maxs[i]; }

    // seq check
    int mn2=a[0], mx2=a[0];
    for(int i=1;i<n;i++){ if(a[i]<mn2) mn2=a[i]; if(a[i]>mx2) mx2=a[i]; }

    printf("[minmax] n=%d  gpu(min=%d max=%d)  seq(min=%d max=%d)\n", n, mn, mx, mn2, mx2);

    free(a); free(mins); free(maxs);
    clReleaseMemObject(inB); clReleaseMemObject(outMinB); clReleaseMemObject(outMaxB);
    clReleaseKernel(k); clReleaseProgram(p);
}

/* ========================= 9) Sliding average ========================= */

void task_sliding_avg(const OclContext* ocl, int n, int radius) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/sliding_avg.cl", "");
    cl_kernel k = ocl_create_kernel(p, "sliding_avg");

    float* a=(float*)malloc(sizeof(float)*(size_t)n);
    float* o=(float*)malloc(sizeof(float)*(size_t)n);
    fill_random_f(a, (size_t)n, 9);

    cl_mem aB = ocl_create_buffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(float)*(size_t)n);
    cl_mem oB = ocl_create_buffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(float)*(size_t)n);
    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, aB, CL_TRUE, 0, sizeof(float)*(size_t)n, a, 0, NULL, NULL));

    OCL_CHECK(clSetKernelArg(k, 0, sizeof(aB), &aB));
    OCL_CHECK(clSetKernelArg(k, 1, sizeof(oB), &oB));
    OCL_CHECK(clSetKernelArg(k, 2, sizeof(int), &n));
    OCL_CHECK(clSetKernelArg(k, 3, sizeof(int), &radius));

    size_t g=(size_t)n;
    OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 0, NULL, NULL));
    OCL_CHECK(clEnqueueReadBuffer(ocl->queue, oB, CL_TRUE, 0, sizeof(float)*(size_t)n, o, 0, NULL, NULL));

    printf("[sliding_avg] n=%d radius=%d first 8:\n", n, radius);
    for (int i=0;i< (n<8?n:8); i++) printf("  i=%d in=%f avg=%f\n", i, a[i], o[i]);

    free(a); free(o);
    clReleaseMemObject(aB); clReleaseMemObject(oB);
    clReleaseKernel(k); clReleaseProgram(p);
}

/* ========================= 10) Prime test ========================= */

void task_prime_test(const OclContext* ocl, uint32_t x) {
    cl_program p = ocl_build_program_from_file(ocl, "kernels/prime_test.cl", "");
    cl_kernel k1 = ocl_create_kernel(p, "prime_one_divisor_per_workitem");
    cl_kernel k2 = ocl_create_kernel(p, "prime_divisor_ranges");
    cl_kernel k3 = ocl_create_kernel(p, "prime_preassigned_primes");

    cl_mem flagB = ocl_create_buffer(ocl->context, CL_MEM_READ_WRITE, sizeof(cl_int));
    cl_int flag = 1;
    OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, flagB, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL));

    uint32_t limit = (uint32_t)floor(sqrt((double)x));
    if (x < 2) { printf("[prime] %u is NOT prime\n", x); goto cleanup; }
    if (x == 2 || x == 3) { printf("[prime] %u is prime\n", x); goto cleanup; }
    if ((x % 2) == 0) { printf("[prime] %u is NOT prime (even)\n", x); goto cleanup; }

    // (a) one divisor per work-item, odd divisors only
    {
        flag = 1;
        OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, flagB, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL));

        OCL_CHECK(clSetKernelArg(k1, 0, sizeof(cl_uint), &x));
        OCL_CHECK(clSetKernelArg(k1, 1, sizeof(cl_uint), &limit));
        OCL_CHECK(clSetKernelArg(k1, 2, sizeof(flagB), &flagB));

        // divs: 3,5,7,...,limit  => count approx (limit-1)/2
        size_t count = (limit >= 3) ? ((size_t)(limit - 1) / 2) : 0;
        if (count == 0) count = 1;
        OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k1, 1, NULL, &count, NULL, 0, NULL, NULL));
        OCL_CHECK(clEnqueueReadBuffer(ocl->queue, flagB, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL));
        printf("[prime one-divisor] %u => %s\n", x, flag ? "prime" : "NOT prime");
    }

    // (b) range per work-item
    {
        flag = 1;
        OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, flagB, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL));

        OCL_CHECK(clSetKernelArg(k2, 0, sizeof(cl_uint), &x));
        OCL_CHECK(clSetKernelArg(k2, 1, sizeof(cl_uint), &limit));
        OCL_CHECK(clSetKernelArg(k2, 2, sizeof(flagB), &flagB));

        size_t workers = 256; // pl. fix
        OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k2, 1, NULL, &workers, NULL, 0, NULL, NULL));
        OCL_CHECK(clEnqueueReadBuffer(ocl->queue, flagB, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL));
        printf("[prime ranges] %u => %s\n", x, flag ? "prime" : "NOT prime");
    }

    // (c) preassigned primes (kis prime lista előre kiosztva)
    {
        flag = 1;
        OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, flagB, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL));

        // small prime list (odd primes only)
        const uint32_t primes[] = {
            3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
            101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199
        };
        const int pc = (int)(sizeof(primes)/sizeof(primes[0]));
        cl_mem pB = ocl_create_buffer(ocl->context, CL_MEM_READ_ONLY, sizeof(cl_uint)*(size_t)pc);
        OCL_CHECK(clEnqueueWriteBuffer(ocl->queue, pB, CL_TRUE, 0, sizeof(cl_uint)*(size_t)pc, primes, 0, NULL, NULL));

        OCL_CHECK(clSetKernelArg(k3, 0, sizeof(cl_uint), &x));
        OCL_CHECK(clSetKernelArg(k3, 1, sizeof(cl_uint), &limit));
        OCL_CHECK(clSetKernelArg(k3, 2, sizeof(pB), &pB));
        OCL_CHECK(clSetKernelArg(k3, 3, sizeof(cl_int), &pc));
        OCL_CHECK(clSetKernelArg(k3, 4, sizeof(flagB), &flagB));

        size_t g=(size_t)pc;
        OCL_CHECK(clEnqueueNDRangeKernel(ocl->queue, k3, 1, NULL, &g, NULL, 0, NULL, NULL));
        OCL_CHECK(clEnqueueReadBuffer(ocl->queue, flagB, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL));
        printf("[prime preassigned primes] %u => %s (only checks listed primes <= sqrt)\n", x, flag ? "prime-ish" : "NOT prime");

        clReleaseMemObject(pB);
    }

cleanup:
    clReleaseMemObject(flagB);
    clReleaseKernel(k1); clReleaseKernel(k2); clReleaseKernel(k3);
    clReleaseProgram(p);
}