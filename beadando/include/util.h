#ifndef UTIL_H
#define UTIL_H

/*
 * Utility module
 *
 * Provides helper functions used across the project,
 * such as high-resolution timing for benchmarking.
 */

/*
 * Returns the current time in milliseconds.
 * Uses a high-resolution timer for accurate performance measurements.
 */
double get_time_ms(void);

#endif