// Tile-olt (lokális mem) mátrixszorzás: C = A(RxK) * B(KxC)
// TILE méretet build optionnal adjuk: -DTILE=16
#ifndef TILE
#define TILE 16
#endif

__kernel void mat_mul_tiled(__global const float* A,
                            __global const float* B,
                            __global float* Cc,
                            int R, int K, int C) {
    int row = (int)get_global_id(0);
    int col = (int)get_global_id(1);

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float sum = 0.0f;

    int lrow = (int)get_local_id(0);
    int lcol = (int)get_local_id(1);

    for (int t0 = 0; t0 < K; t0 += TILE) {
        int aCol = t0 + lcol;
        int bRow = t0 + lrow;

        As[lrow][lcol] = (row < R && aCol < K) ? A[row*K + aCol] : 0.0f;
        Bs[lrow][lcol] = (bRow < K && col < C) ? B[bRow*C + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0;k<TILE;k++) sum += As[lrow][k] * Bs[k][lcol];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < R && col < C) Cc[row*C + col] = sum;
}

// Tile-olt transzponálás (bank konfliktus csökkentés: TILE x (TILE+1))
__kernel void mat_transpose_tiled(__global const float* A,
                                  __global float* AT,
                                  int R, int C) {
    __local float tile[TILE][TILE+1];

    int r = (int)get_global_id(0);
    int c = (int)get_global_id(1);
    int lr = (int)get_local_id(0);
    int lc = (int)get_local_id(1);

    if (r < R && c < C) tile[lr][lc] = A[r*C + c];
    barrier(CLK_LOCAL_MEM_FENCE);

    int r2 = (int)(get_group_id(1) * TILE + lr);
    int c2 = (int)(get_group_id(0) * TILE + lc);
    if (r2 < C && c2 < R) AT[r2*R + c2] = tile[lc][lr];
}