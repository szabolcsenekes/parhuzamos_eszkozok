__kernel void map_global_index(__global int* out, int n) {
    int gid = (int)get_global_id(0);
    if (gid < n) out[gid] = gid;
}

__kernel void map_local_index(__global int* out, int n) {
    int gid = (int)get_global_id(0);
    int lid = (int)get_local_id(0);
    if (gid < n) out[gid] = lid;
}