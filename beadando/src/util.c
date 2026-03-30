#include <windows.h>
#include "util.h"

/*
 * Returns the current time in milliseconds using a high-resolution timer.
 *
 * This function uses Windows' QueryPerformanceCounter API, which provides
 * much higher precision than standard timers like clock() or time().
 *
 * The frequency of the performance counter is queried only once and stored
 * statically for efficiency.
 */
double get_time_ms(void)
{
    static LARGE_INTEGER frequency;
    static int initialized = 0;

    /*
     * Initialize the frequency only once.
     * This value represents counts per second.
     */
    if (!initialized)
    {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    LARGE_INTEGER counter;

    /* Get the current counter value. */
    QueryPerformanceCounter(&counter);

    /*
     * Convert counter value to milliseconds:
     * (counts / counts_per_second) * 1000
     */
    return (double)(counter.QuadPart * 1000.0 / frequency.QuadPart);
}