__kernel void count_occurrence(__global const int* a, __global int* count, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) return;

    int ai = a[i];
    int c = 0;
    for (int j=0;j<n;j++) {
        if (a[j] == ai) c++;
    }
    count[i] = c;
}