__kernel void rank_all(__global const int* a, __global int* rank, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) return;

    int ai = a[i];
    int r = 0;
    for (int j=0;j<n;j++) {
        if (a[j] < ai) r++;
    }
    rank[i] = r;
}