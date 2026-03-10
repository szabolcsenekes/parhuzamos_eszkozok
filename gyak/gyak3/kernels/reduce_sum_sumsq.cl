__kernel void reduce_sum_sumsq(__global const float* a,
                               __global float* partialSum,
                               __global float* partialSumSq,
                               int n,
                               __local float* lsum,
                               __local float* lsum2) {
    int gid = (int)get_global_id(0);
    int lid = (int)get_local_id(0);
    int lsz = (int)get_local_size(0);
    int grp = (int)get_group_id(0);

    float x = (gid < n) ? a[gid] : 0.0f;
    lsum[lid]  = x;
    lsum2[lid] = x * x;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz/2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            lsum[lid]  += lsum[lid + stride];
            lsum2[lid] += lsum2[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partialSum[grp]   = lsum[0];
        partialSumSq[grp] = lsum2[0];
    }
}