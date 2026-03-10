__kernel void histogram_0_100(__global const int* a,
                              __global int* hist,
                              int n) {
    int i = (int)get_global_id(0);
    if (i >= n) return;
    int v = a[i];
    if ((uint)v <= 100u) {
        atomic_inc((volatile __global int*)&hist[v]);
    }
}