__kernel void map_reverse(__global const int* in, __global int* out, int n) {
    int i = (int)get_global_id(0);
    if (i < n) out[i] = in[n - 1 - i];
}