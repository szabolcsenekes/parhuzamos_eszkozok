#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <SDL2/SDL.h>

#include "renderer.h"
#include "grid.h"

/* SDL objects used internally by the renderer module. */
static SDL_Window *window = NULL;
static SDL_Renderer *renderer = NULL;
static SDL_Texture *texture = NULL;

/*
 * Converts a normalized temperature value in the range [0, 1]
 * to an RGB color for visualization.
 *
 * Lower temperatures are displayed as blue, higher temperatures
 * move toward red, with intermediate values producing a gradient.
 */
static void heat_to_color(float t, Uint8 *r, Uint8 *g, Uint8 *b)
{
    /* Clamp the input temperature to the valid range. */
    if (t < 0.0f)
        t = 0.0f;
    if (t > 1.0f)
        t = 1.0f;

    /*
     * Square root scaling makes low temperature differences
     * more visible on screen.
     */
    float s = sqrtf(t);

    *r = (Uint8)(255.0f * s);
    *g = (Uint8)(255.0f * (1.0f - fabsf(2.0f * s - 1.0f)));
    *b = (Uint8)(255.0f * (1.0f - s));
}

/*
 * Initializes SDL and creates the window, renderer, and texture
 * used for displaying the simulation.
 *
 * The window size is based on the grid dimensions and the
 * selected rendering scale.
 */
void init_sdl(const Grid *g)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        printf("SDL_Init error: %s\n", SDL_GetError());
        exit(1);
    }

    window = SDL_CreateWindow(
        "Heat Simulation",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        g->width * g->window_scale,
        g->height * g->window_scale,
        0);

    if (window == NULL)
    {
        printf("SDL_CreateWindow error: %s\n", SDL_GetError());
        exit(1);
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL)
    {
        printf("SDL_CreateRenderer error: %s\n", SDL_GetError());
        exit(1);
    }

    /*
     * The texture stores the current frame as a 2D RGB image.
     * Its size matches the simulation grid resolution.
     */
    texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        g->width,
        g->height);

    if (texture == NULL)
    {
        printf("SDL_CreateTexture error: %s\n", SDL_GetError());
        exit(1);
    }
}

/*
 * Renders the current simulation state to the SDL window.
 *
 * Each cell of the temperature grid is converted to an RGB color
 * and written into the texture pixel buffer, which is then displayed.
 */
void render(const Grid *g)
{
    Uint8 *pixels;
    int pitch;

    if (SDL_LockTexture(texture, NULL, (void **)&pixels, &pitch) != 0)
    {
        printf("SDL_LockTexture error: %s\n", SDL_GetError());
        return;
    }

    for (int y = 0; y < g->height; y++)
    {
        for (int x = 0; x < g->width; x++)
        {
            int idx = y * g->width + x;
            Uint8 r, g_col, b;

            /* Convert the current temperature value to a display color. */
            heat_to_color(g->current[idx], &r, &g_col, &b);

            int offset = y * pitch + x * 3;
            pixels[offset] = r;
            pixels[offset + 1] = g_col;
            pixels[offset + 2] = b;
        }
    }

    SDL_UnlockTexture(texture);

    /* Clear the window, draw the updated texture, then present it. */
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

/*
 * Releases all SDL resources used by the renderer.
 */
void cleanup_sdl(void)
{
    if (texture != NULL)
        SDL_DestroyTexture(texture);
    if (renderer != NULL)
        SDL_DestroyRenderer(renderer);
    if (window != NULL)
        SDL_DestroyWindow(window);

    SDL_Quit();

    texture = NULL;
    renderer = NULL;
    window = NULL;
}