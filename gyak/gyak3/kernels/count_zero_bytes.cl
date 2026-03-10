__kernel void count_zero_bytes(__global const uchar* data,
                               int nbytes,
                               __global uint* groupCounts,
                               __local uint* lcnt) {
    int gid = (int)get_global_id(0);
    int lid = (int)get_local_id(0);
    int lsz = (int)get_local_size(0);
    int grp = (int)get_group_id(0);

    uint c = 0;

    // 16 bájtos vektoros feldolgozás
    int idx = gid * 16;
    if (idx + 15 < nbytes) {
        uchar16 v = vload16(0, data + idx);
        // összehasonlítás 0-val
        ushort16 m = convert_ushort16(v == (uchar)0);
        // m elemei 0 vagy 1 (implementációfüggően), de jó gyakorlat: csak !=0-t számoljunk
        c += (m.s0  != 0) + (m.s1  != 0) + (m.s2  != 0) + (m.s3  != 0);
        c += (m.s4  != 0) + (m.s5  != 0) + (m.s6  != 0) + (m.s7  != 0);
        c += (m.s8  != 0) + (m.s9  != 0) + (m.sa  != 0) + (m.sb  != 0);
        c += (m.sc  != 0) + (m.sd  != 0) + (m.se  != 0) + (m.sf  != 0);
    } else {
        // maradék
        for (int j=idx; j<nbytes && j<idx+16; j++) {
            c += (data[j] == (uchar)0);
        }
    }

    lcnt[lid] = c;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz/2; stride>0; stride >>= 1) {
        if (lid < stride) lcnt[lid] += lcnt[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) groupCounts[grp] = lcnt[0];
}