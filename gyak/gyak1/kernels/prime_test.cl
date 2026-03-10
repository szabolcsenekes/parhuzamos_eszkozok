inline void set_not_prime(__global int* flag) {
    // 1 => prime until proven otherwise
    // 0 => not prime
    atomic_xchg((volatile __global int*)flag, 0);
}

// 1) Minden work-item 1 osztót vizsgál (odd divisors: 3,5,7,...)
__kernel void prime_one_divisor_per_workitem(uint x, uint limit, __global int* flag) {
    size_t idx = get_global_id(0);
    if (*flag == 0) return;

    uint d = 3u + 2u*(uint)idx;
    if (d > limit) return;
    if ((x % d) == 0u) set_not_prime(flag);
}

// 2) Work-itemhez osztótartomány tartozik
__kernel void prime_divisor_ranges(uint x, uint limit, __global int* flag) {
    size_t gid = get_global_id(0);
    size_t gsz = get_global_size(0);
    if (*flag == 0) return;

    // odd divisors count
    uint count = (limit >= 3u) ? ((limit - 1u) / 2u) : 0u; // 3..limit step2
    if (count == 0u) return;

    uint chunk = (count + (uint)gsz - 1u) / (uint)gsz;
    uint start = (uint)gid * chunk;
    uint end   = start + chunk;
    if (start >= count) return;
    if (end > count) end = count;

    for (uint k = start; k < end; k++) {
        if (*flag == 0) return;
        uint d = 3u + 2u*k;
        if ((x % d) == 0u) { set_not_prime(flag); return; }
    }
}

// 3) Előre kiosztott prímekkel (csak ezeket osztja)
__kernel void prime_preassigned_primes(uint x, uint limit,
                                       __global const uint* primes, int primeCount,
                                       __global int* flag) {
    int i = (int)get_global_id(0);
    if (i >= primeCount) return;
    if (*flag == 0) return;

    uint p = primes[i];
    if (p > limit) return;
    if ((x % p) == 0u) set_not_prime(flag);
}