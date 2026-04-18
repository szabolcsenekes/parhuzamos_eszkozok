#ifndef GRID_H
#define GRID_H

/*
 * Grid module
 *
 * Encapsulates the simulation state, including dimensions,
 * rendering scale, temperature buffers, and heat source map.
 */

typedef struct Grid
{
    int width;
    int height;
    int window_scale;

    float *current;
    float *next;
    unsigned char *source_map;
} Grid;

/* Initializes the grid with the given size and scale. */
void init_grid(Grid *g, int width, int height, int scale);

/* Resets the grid to the initial state. */
void reset_grid(Grid *g);

/* Frees all memory associated with the grid. */
void free_grid(Grid *g);

/* Adds a permanent heat source at the given grid position. */
void add_heat_source(Grid *g, int mouse_x, int mouse_y, int radius);

#endif