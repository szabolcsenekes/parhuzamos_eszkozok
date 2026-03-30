#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include "grid.h"
#include "renderer.h"
#include "util.h"
#include "benchmark.h"
#include "opencl_heat.h"

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    run_multi_size_benchmarks("data/output.csv");

    init_grid(512, 512, 2);
    init_sdl();
    init_opencl();
    reset_grid();

    bool running = true;
    bool paused = false;
    SDL_Event event;

    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
                running = false;

            if (event.type == SDL_MOUSEBUTTONDOWN &&
                event.button.button == SDL_BUTTON_LEFT)
            {
                int mouse_x = event.button.x / window_scale;
                int mouse_y = event.button.y / window_scale;
                add_heat_source(mouse_x, mouse_y, 6);
            }

            if (event.type == SDL_KEYDOWN)
            {
                switch (event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    running = false;
                    break;
                case SDLK_SPACE:
                    paused = !paused;
                    break;
                case SDLK_r:
                    reset_grid();
                    break;
                }
            }
        }

        if (!paused)
        {
            run_kernel();
        }

        render();
        SDL_Delay(16);
    }

    cleanup_opencl();
    cleanup_sdl();
    free_grid();

    return 0;
}