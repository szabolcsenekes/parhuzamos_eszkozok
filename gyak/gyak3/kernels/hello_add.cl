__kernel void hello_add(__global const float* a,
                        __global const float* b,
                        __global float* out,
                        int n) {
    int i = (int)get_global_id(0);
    if (i < n) out[i] = a[i] + b[i];
}