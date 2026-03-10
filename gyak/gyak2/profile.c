#include "profile.h"
#include <stdio.h>

double event_ms(cl_event ev) {
    cl_ulong t0=0, t1=0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (double)(t1 - t0) / 1e6; // ns -> ms
}

void write_csv_hello_profile(const char* path,
                             double t_write_ms,
                             double t_kernel_ms,
                             double t_read_ms) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "step,ms\n");
    fprintf(f, "write,%.6f\n", t_write_ms);
    fprintf(f, "kernel,%.6f\n", t_kernel_ms);
    fprintf(f, "read,%.6f\n", t_read_ms);
    fclose(f);
}