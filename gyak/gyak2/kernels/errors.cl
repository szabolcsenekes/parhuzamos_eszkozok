// 0-val osztás float (várhatóan inf/nan)
__kernel void div0_float(__global float* out) {
    int i = get_global_id(0);
    float a = 1.0f;
    float z = 0.0f;
    out[i] = a / z;
}

// 0-val osztás int (undefined / driverfüggő, demonstráció)
__kernel void div0_int(__global int* out) {
    int i = get_global_id(0);
    int a = 1;
    int z = 0;
    out[i] = a / z;
}

// OpenCL C limitáció: REKURZIÓ (build error)
int rec(int x) { return x <= 0 ? 0 : 1 + rec(x-1); }
__kernel void bad_recursion(__global int* out) {
    out[0] = rec(10);
}