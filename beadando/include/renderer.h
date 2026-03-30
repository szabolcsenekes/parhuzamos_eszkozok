#ifndef RENDERER_H
#define RENDERER_H

#include <SDL2/SDL.h>

/*
 * Renderer module
 *
 * Handles visualization of the heat simulation using SDL2.
 * Responsible for window creation, drawing the grid,
 * and cleaning up rendering resources.
 */

/* SDL objects used for rendering. */
extern SDL_Window *window;
extern SDL_Renderer *renderer;
extern SDL_Texture *texture;

/* Initializes SDL, creates window, renderer, and texture. */
void init_sdl(void);

/* Renders the current simulation state to the window. */
void render(void);

/* Releases all SDL-related resources. */
void cleanup_sdl(void);

#endif