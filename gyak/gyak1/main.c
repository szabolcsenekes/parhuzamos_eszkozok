#include "ocl_utils.h"
#include "task.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* exe) {
    printf("Usage:\n");
    printf("  %s info\n", exe);
    printf("  %s map_index [n] [global|local]\n", exe);
    printf("  %s map_reverse [n]\n", exe);
    printf("  %s map_swap [n]\n", exe);
    printf("  %s vec_add [n]\n", exe);
    printf("  %s fill_missing [n]\n", exe);
    printf("  %s rank [n]\n", exe);
    printf("  %s occurrence [n]\n", exe);
    printf("  %s minmax [n]\n", exe);
    printf("  %s sliding_avg [n] [radius]\n", exe);
    printf("  %s prime [x]\n", exe);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 0; }

    if (strcmp(argv[1], "info") == 0) {
        ocl_print_device_info_like_clinfo();
        return 0;
    }

    OclContext ocl;
    if (!ocl_init_default(&ocl)) {
        printf("Failed to init OpenCL (no platform/device?).\n");
        return 1;
    }

    if (strcmp(argv[1], "map_index") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 64;
        int local = (argc > 3 && strcmp(argv[3], "local")==0);
        task_map_index(&ocl, local, n);
    } else if (strcmp(argv[1], "map_reverse") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 64;
        task_map_reverse(&ocl, n);
    } else if (strcmp(argv[1], "map_swap") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 64;
        task_map_swap_neighbors(&ocl, n);
    } else if (strcmp(argv[1], "vec_add") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 1<<20;
        task_vec_add_demo(&ocl, n);
    } else if (strcmp(argv[1], "fill_missing") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 256;
        task_fill_missing(&ocl, n);
    } else if (strcmp(argv[1], "rank") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 256;
        task_rank(&ocl, n);
    } else if (strcmp(argv[1], "occurrence") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 256;
        task_occurrence(&ocl, n);
    } else if (strcmp(argv[1], "minmax") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 1<<20;
        task_minmax(&ocl, n);
    } else if (strcmp(argv[1], "sliding_avg") == 0) {
        int n = (argc > 2) ? atoi(argv[2]) : 1024;
        int r = (argc > 3) ? atoi(argv[3]) : 3;
        task_sliding_avg(&ocl, n, r);
    } else if (strcmp(argv[1], "prime") == 0) {
        unsigned x = (argc > 2) ? (unsigned)strtoul(argv[2], NULL, 10) : 2147483647u;
        task_prime_test(&ocl, (uint32_t)x);
    } else {
        usage(argv[0]);
    }

    ocl_cleanup(&ocl);
    return 0;
}