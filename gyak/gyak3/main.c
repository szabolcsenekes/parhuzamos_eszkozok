#include "ocl_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double now_ms(void){
    static LARGE_INTEGER freq;
    static int init=0;
    if(!init){ QueryPerformanceFrequency(&freq); init=1; }
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double now_ms(void){
    struct timeval tv; gettimeofday(&tv,NULL);
    return (double)tv.tv_sec*1000.0 + (double)tv.tv_usec/1000.0;
}
#endif

static void die_build(const BuildResult* br, const char* where){
    if (br->err != CL_SUCCESS) {
        fprintf(stderr, "[%s] build error %d (%s)\n", where, br->err, cl_error_name(br->err));
        if (br->build_log) fprintf(stderr, "Build log:\n%s\n", br->build_log);
        exit(1);
    }
}

static void usage(const char* exe){
    printf("Usage:\n");
    printf("  %s event_demo\n", exe);
    printf("  %s histogram N\n", exe);
    printf("  %s stddev N\n", exe);
    printf("  %s make_file path sizeMB\n", exe);
    printf("  %s count_zeros path chunkMB\n", exe);
}

/* ================== 1) EVENT + CALLBACK DEMO ================== */

typedef struct {
    float* host_out;
    int n;
    cl_event kernel_ev;
    cl_event read_ev;
} ReadCallbackData;

static void CL_CALLBACK read_complete_cb(cl_event ev, cl_int status, void* user_data) {
    (void)ev;
    ReadCallbackData* d = (ReadCallbackData*)user_data;
    printf("[callback] read status=%d (%s)\n", status, cl_error_name(status));

    // Kernel event ideje (profil)
    double k_ms = event_ms(d->kernel_ev);
    double r_ms = event_ms(d->read_ev);
    printf("[callback] kernel=%.3fms read=%.3fms\n", k_ms, r_ms);

    // Mutassunk pár elemet a kiolvasott bufferből
    int show = d->n < 8 ? d->n : 8;
    for(int i=0;i<show;i++) {
        printf("  out[%d]=%f\n", i, d->host_out[i]);
    }
}

static void event_demo(const Ocl* ocl){
    const int N = 1024;

    BuildResult br = build_program_from_file(ocl, "kernels/hello_add.cl", "");
    die_build(&br, "event_demo");
    cl_kernel k = kernel_or_die(br.program, "hello_add");

    float* a = (float*)malloc(sizeof(float)*N);
    float* b = (float*)malloc(sizeof(float)*N);
    float* out = (float*)malloc(sizeof(float)*N);
    for(int i=0;i<N;i++){ a[i]=(float)i; b[i]=1.0f; out[i]=0.0f; }

    cl_int err=0;
    cl_mem aB = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(float)*N, NULL, &err);
    cl_mem bB = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(float)*N, NULL, &err);
    cl_mem oB = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(float)*N, NULL, &err);

    cl_event evW1, evW2, evK, evR;
    clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*N, a, 0, NULL, &evW1);
    clEnqueueWriteBuffer(ocl->queue, bB, CL_FALSE, 0, sizeof(float)*N, b, 0, NULL, &evW2);

    clSetKernelArg(k,0,sizeof(aB),&aB);
    clSetKernelArg(k,1,sizeof(bB),&bB);
    clSetKernelArg(k,2,sizeof(oB),&oB);
    clSetKernelArg(k,3,sizeof(int),&N);

    size_t g=(size_t)N;
    cl_event deps[2]={evW1, evW2};
    clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 2, deps, &evK);

    // Aszinkron read + callback
    clEnqueueReadBuffer(ocl->queue, oB, CL_FALSE, 0, sizeof(float)*N, out, 1, &evK, &evR);

    ReadCallbackData cb;
    cb.host_out = out;
    cb.n = N;
    cb.kernel_ev = evK;
    cb.read_ev = evR;

    // callback a read eventre (amikor kész, out[] már tele van)
    clSetEventCallback(evR, CL_COMPLETE, read_complete_cb, &cb);

    // biztosítsuk, hogy a callback lefusson (eventek lefutnak)
    clFinish(ocl->queue);

    clReleaseEvent(evW1); clReleaseEvent(evW2); clReleaseEvent(evK); clReleaseEvent(evR);
    clReleaseMemObject(aB); clReleaseMemObject(bB); clReleaseMemObject(oB);
    clReleaseKernel(k);
    free_build_result(&br);
    free(a); free(b); free(out);
}

/* ================== 2) HISTOGRAM [0..100] ================== */

static void histogram_demo(const Ocl* ocl, int N){
    BuildResult br = build_program_from_file(ocl, "kernels/histogram_0_100.cl", "");
    die_build(&br, "histogram");
    cl_kernel k = kernel_or_die(br.program, "histogram_0_100");

    int* a = (int*)malloc(sizeof(int)* (size_t)N);
    for(int i=0;i<N;i++) a[i] = rand() % 101;

    int hist[101]; memset(hist, 0, sizeof(hist));

    cl_int err=0;
    cl_mem aB = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,  sizeof(int)*(size_t)N, NULL, &err);
    cl_mem hB = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(int)*101, NULL, &err);

    cl_event evW1, evW2, evK, evR;
    clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(int)*(size_t)N, a, 0, NULL, &evW1);
    clEnqueueWriteBuffer(ocl->queue, hB, CL_FALSE, 0, sizeof(int)*101, hist, 0, NULL, &evW2);

    clSetKernelArg(k,0,sizeof(aB),&aB);
    clSetKernelArg(k,1,sizeof(hB),&hB);
    clSetKernelArg(k,2,sizeof(int),&N);

    size_t g=(size_t)N;
    cl_event deps[2]={evW1, evW2};
    clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &g, NULL, 2, deps, &evK);
    clEnqueueReadBuffer(ocl->queue, hB, CL_TRUE, 0, sizeof(int)*101, hist, 1, &evK, &evR);

    // ellenőrzés: összegek
    long long sum=0;
    for(int i=0;i<=100;i++) sum += hist[i];

    printf("[histogram] N=%d sum(hist)=%lld | write=%.3fms kernel=%.3fms read=%.3fms\n",
           N, sum, event_ms(evW1)+event_ms(evW2), event_ms(evK), event_ms(evR));
    printf("  hist[0]=%d hist[1]=%d hist[100]=%d\n", hist[0], hist[1], hist[100]);

    clReleaseEvent(evW1); clReleaseEvent(evW2); clReleaseEvent(evK); clReleaseEvent(evR);
    clReleaseMemObject(aB); clReleaseMemObject(hB);
    clReleaseKernel(k);
    free_build_result(&br);
    free(a);
}

/* ================== 3) STDDEV ================== */

static void stddev_demo(const Ocl* ocl, int N){
    BuildResult br = build_program_from_file(ocl, "kernels/reduce_sum_sumsq.cl", "");
    die_build(&br, "stddev");
    cl_kernel k = kernel_or_die(br.program, "reduce_sum_sumsq");

    float* a = (float*)malloc(sizeof(float)*(size_t)N);
    for(int i=0;i<N;i++) a[i] = (float)(rand()%10000) / 100.0f;

    size_t local = 256;
    size_t groups = ((size_t)N + local - 1) / local;
    size_t global = groups * local;

    cl_int err=0;
    cl_mem aB = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(float)*(size_t)N, NULL, &err);
    cl_mem pS = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(float)*groups, NULL, &err);
    cl_mem pQ = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(float)*groups, NULL, &err);

    cl_event evW, evK, evR1, evR2;
    clEnqueueWriteBuffer(ocl->queue, aB, CL_FALSE, 0, sizeof(float)*(size_t)N, a, 0, NULL, &evW);

    clSetKernelArg(k,0,sizeof(aB),&aB);
    clSetKernelArg(k,1,sizeof(pS),&pS);
    clSetKernelArg(k,2,sizeof(pQ),&pQ);
    clSetKernelArg(k,3,sizeof(int),&N);
    clSetKernelArg(k,4,sizeof(float)*local,NULL);
    clSetKernelArg(k,5,sizeof(float)*local,NULL);

    clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &global, &local, 1, &evW, &evK);

    float* hs = (float*)malloc(sizeof(float)*groups);
    float* hq = (float*)malloc(sizeof(float)*groups);
    clEnqueueReadBuffer(ocl->queue, pS, CL_FALSE, 0, sizeof(float)*groups, hs, 1, &evK, &evR1);
    clEnqueueReadBuffer(ocl->queue, pQ, CL_TRUE,  0, sizeof(float)*groups, hq, 1, &evK, &evR2);

    double sum=0.0, sumsq=0.0;
    for(size_t i=0;i<groups;i++){ sum += hs[i]; sumsq += hq[i]; }

    double mean = sum / (double)N;
    double var  = sumsq / (double)N - mean*mean;
    if (var < 0) var = 0; // numerikus védelem
    double sd = sqrt(var);

    printf("[stddev] N=%d mean=%g stddev=%g | write=%.3fms kernel=%.3fms read=%.3fms\n",
           N, mean, sd, event_ms(evW), event_ms(evK), event_ms(evR1)+event_ms(evR2));

    clReleaseEvent(evW); clReleaseEvent(evK); clReleaseEvent(evR1); clReleaseEvent(evR2);
    clReleaseMemObject(aB); clReleaseMemObject(pS); clReleaseMemObject(pQ);
    clReleaseKernel(k);
    free_build_result(&br);
    free(a); free(hs); free(hq);
}

/* ================== 4) BIG FILE + ZERO BYTE COUNT ================== */

static int make_file(const char* path, int sizeMB){
    FILE* f = fopen(path, "wb");
    if(!f){ printf("Cannot create %s\n", path); return 0; }

    const size_t chunk = 8u * 1024u * 1024u;
    unsigned char* buf = (unsigned char*)malloc(chunk);
    if(!buf){ fclose(f); return 0; }

    // generálás: legyen benne sok 0 is (pl. 1/16 arány)
    for(int mb=0; mb<sizeMB; ){
        size_t to = chunk;
        int remainMB = sizeMB - mb;
        if ((size_t)remainMB * 1024u * 1024u < to) to = (size_t)remainMB * 1024u * 1024u;

        for(size_t i=0;i<to;i++){
            int r = rand() & 15;
            buf[i] = (r==0) ? 0 : (unsigned char)(rand() & 255);
        }
        fwrite(buf, 1, to, f);
        mb += (int)(to / (1024u*1024u));
        if (to % (1024u*1024u)) mb += 1; // durva közelítés, nem számít
    }

    free(buf);
    fclose(f);
    printf("Created %s (%d MB)\n", path, sizeMB);
    return 1;
}

static void count_zeros_file(const Ocl* ocl, const char* path, int chunkMB){
    BuildResult br = build_program_from_file(ocl, "kernels/count_zero_bytes.cl", "");
    die_build(&br, "count_zeros");
    cl_kernel k = kernel_or_die(br.program, "count_zero_bytes");

    FILE* f = fopen(path, "rb");
    if(!f){ printf("Cannot open %s\n", path); exit(1); }

    const size_t chunkBytes = (size_t)chunkMB * 1024u * 1024u;
    unsigned char* hostBuf = (unsigned char*)malloc(chunkBytes);
    if(!hostBuf){ printf("malloc failed\n"); exit(1); }

    // Device input buffer: újrahasznosítjuk chunkonként
    cl_int err=0;
    cl_mem inB = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, chunkBytes, NULL, &err);
    if(err!=CL_SUCCESS){ printf("clCreateBuffer failed\n"); exit(1); }

    FILE* csv = fopen("zero_count_profile.csv", "wb");
    if(csv) fprintf(csv, "chunk_index,bytes,read_file_ms,write_ms,kernel_ms,read_ms,total_ms,zeros\n");

    unsigned long long totalZeros = 0;
    unsigned long long totalBytes = 0;
    int chunkIdx = 0;

    while(1){
        double t_file0 = now_ms();
        size_t rd = fread(hostBuf, 1, chunkBytes, f);
        double t_file1 = now_ms();
        if(rd == 0) break;

        totalBytes += rd;

        // Kernel: nbytes=rd, global threads = ceil((rd/16)/local)*local
        int nbytes = (int)rd;
        size_t items = (size_t)((nbytes + 15) / 16); // 1 work-item ~ 16 bytes
        size_t local = 256;
        size_t groups = (items + local - 1) / local;
        size_t global = groups * local;

        cl_mem groupB = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, sizeof(cl_uint)*groups, NULL, &err);

        cl_event evW, evK, evR;
        clEnqueueWriteBuffer(ocl->queue, inB, CL_FALSE, 0, rd, hostBuf, 0, NULL, &evW);

        clSetKernelArg(k,0,sizeof(inB),&inB);
        clSetKernelArg(k,1,sizeof(int),&nbytes);
        clSetKernelArg(k,2,sizeof(groupB),&groupB);
        clSetKernelArg(k,3,sizeof(cl_uint)*local,NULL);

        clEnqueueNDRangeKernel(ocl->queue, k, 1, NULL, &global, &local, 1, &evW, &evK);

        cl_uint* groupsOut = (cl_uint*)malloc(sizeof(cl_uint)*groups);
        clEnqueueReadBuffer(ocl->queue, groupB, CL_TRUE, 0, sizeof(cl_uint)*groups, groupsOut, 1, &evK, &evR);

        unsigned long long chunkZeros = 0;
        for(size_t i=0;i<groups;i++) chunkZeros += groupsOut[i];
        totalZeros += chunkZeros;

        double w_ms = event_ms(evW);
        double k_ms = event_ms(evK);
        double r_ms = event_ms(evR);
        double total_ms = (t_file1 - t_file0) + w_ms + k_ms + r_ms;

        printf("[chunk %d] bytes=%zu file=%.3fms write=%.3fms kernel=%.3fms read=%.3fms zeros=%llu\n",
               chunkIdx, rd, (t_file1 - t_file0), w_ms, k_ms, r_ms, chunkZeros);

        if(csv) fprintf(csv, "%d,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%llu\n",
                        chunkIdx, rd, (t_file1 - t_file0), w_ms, k_ms, r_ms, total_ms, chunkZeros);

        free(groupsOut);
        clReleaseEvent(evW); clReleaseEvent(evK); clReleaseEvent(evR);
        clReleaseMemObject(groupB);

        chunkIdx++;
    }

    if(csv){ fclose(csv); printf("Wrote zero_count_profile.csv\n"); }
    printf("[TOTAL] bytes=%llu zeros=%llu\n", totalBytes, totalZeros);

    clReleaseMemObject(inB);
    clReleaseKernel(k);
    free_build_result(&br);
    free(hostBuf);
    fclose(f);
}

int main(int argc, char** argv){
    if(argc < 2){ usage(argv[0]); return 0; }

    Ocl ocl;
    if(!ocl_init(&ocl)){
        printf("OpenCL init failed\n");
        return 1;
    }

    if(strcmp(argv[1],"event_demo")==0){
        event_demo(&ocl);
    } else if(strcmp(argv[1],"histogram")==0){
        int N = (argc>=3) ? atoi(argv[2]) : 1000000;
        histogram_demo(&ocl, N);
    } else if(strcmp(argv[1],"stddev")==0){
        int N = (argc>=3) ? atoi(argv[2]) : 1000000;
        stddev_demo(&ocl, N);
    } else if(strcmp(argv[1],"make_file")==0){
        if(argc<4){ usage(argv[0]); }
        else make_file(argv[2], atoi(argv[3]));
    } else if(strcmp(argv[1],"count_zeros")==0){
        if(argc<4){ usage(argv[0]); }
        else count_zeros_file(&ocl, argv[2], atoi(argv[3]));
    } else {
        usage(argv[0]);
    }

    ocl_cleanup(&ocl);
    return 0;
}