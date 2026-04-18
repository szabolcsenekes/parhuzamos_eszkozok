#include <SDL2/SDL.h>
#include "util.h"

/*
 * Returns the current time in milliseconds using SDL's high-resolution timer.
 *
 * This implementation is platform-independent and works on Windows,
 * Linux, and macOS as long as SDL2 is available.
 */
double get_time_ms(void)
{
    Uint64 counter = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();

    if (frequency == 0)
    {
        return 0.0;
    }

    return (double)counter * 1000.0 / (double)frequency;
}