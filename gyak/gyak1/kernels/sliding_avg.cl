__kernel void sliding_avg(__global const float* in,
                          __global float* out,
                          int n,
                          int radius) {
    int i = (int)get_global_id(0);
    if (i >= n) return;

    int lo = i - radius; if (lo < 0) lo = 0;
    int hi = i + radius; if (hi >= n) hi = n-1;

    float sum = 0.0f;
    int cnt = 0;
    for (int j=lo; j<=hi; j++) { sum += in[j]; cnt++; }
    out[i] = sum / (float)cnt;
}