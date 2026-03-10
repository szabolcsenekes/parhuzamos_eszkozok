__kernel void swap_neighbors(__global const int* in, __global int* out, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) return;

    // páros<->páratlan csere: 0<->1, 2<->3, ...
    int j;
    if ((i & 1) == 0) j = i + 1;
    else              j = i - 1;

    if (j >= 0 && j < n) out[i] = in[j];
    else out[i] = in[i]; // ha n páratlan, az utolsó magára marad
}