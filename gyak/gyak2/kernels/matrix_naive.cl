__kernel void mat_transpose(__global const float* A,
                            __global float* AT,
                            int R, int C) {
    int r = (int)get_global_id(0);
    int c = (int)get_global_id(1);
    if (r < R && c < C) {
        AT[c*R + r] = A[r*C + c];
    }
}

__kernel void mat_mul_naive(__global const float* A,
                            __global const float* B,
                            __global float* Cc,
                            int R, int K, int C) {
    int r = (int)get_global_id(0);
    int c = (int)get_global_id(1);
    if (r < R && c < C) {
        float sum = 0.0f;
        for (int t=0;t<K;t++) sum += A[r*K + t] * B[t*C + c];
        Cc[r*C + c] = sum;
    }
}

__kernel void row_sum(__global const float* A, __global float* out, int R, int C) {
    int r = (int)get_global_id(0);
    if (r < R) {
        float s=0.0f;
        for (int c=0;c<C;c++) s += A[r*C + c];
        out[r] = s;
    }
}

__kernel void col_sum(__global const float* A, __global float* out, int R, int C) {
    int c = (int)get_global_id(0);
    if (c < C) {
        float s=0.0f;
        for (int r=0;r<R;r++) s += A[r*C + c];
        out[c] = s;
    }
}