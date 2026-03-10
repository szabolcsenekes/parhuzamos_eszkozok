__kernel void fill_missing(__global uint* a,
                           __global const uchar* missing,
                           int n) {
    int i = (int)get_global_id(0);
    if (i <= 0 || i >= n-1) return;
    if (missing[i]) {
        uint left = a[i-1];
        uint right = a[i+1];
        a[i] = (left + right) / 2;
    }
}