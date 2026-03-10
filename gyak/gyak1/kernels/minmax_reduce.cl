__kernel void minmax_reduce_pass(__global const int* in,
                                 __global int* outMin,
                                 __global int* outMax,
                                 int n,
                                 __local int* sMin,
                                 __local int* sMax) {
    int gid = (int)get_global_id(0);
    int lid = (int)get_local_id(0);
    int lsz = (int)get_local_size(0);
    int grp = (int)get_group_id(0);

    int v = (gid < n) ? in[gid] : 2147483647;
    int w = (gid < n) ? in[gid] : -2147483648;

    sMin[lid] = v;
    sMax[lid] = w;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz/2; stride>0; stride/=2) {
        if (lid < stride) {
            int a = sMin[lid];
            int b = sMin[lid + stride];
            sMin[lid] = (a < b) ? a : b;

            int c = sMax[lid];
            int d = sMax[lid + stride];
            sMax[lid] = (c > d) ? c : d;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        outMin[grp] = sMin[0];
        outMax[grp] = sMax[0];
    }
}