#ifndef RENDERER_H
#define RENDERER_H

#include <SDL2/SDL.h>
#include "grid.h"

/*
 * Renderer module
 *
 * Handles visualization of the heat simulation using SDL2.
 * Responsible for window creation, drawing the grid,
 * and cleaning up rendering resources.
 */

/* Initializes SDL, creates window, renderer, and texture. */
void init_sdl(const Grid *g);

/* Renders the current simulation state to the window. */
void render(const Grid *g);

/* Releases all SDL-related resources. */
void cleanup_sdl(void);

#endif