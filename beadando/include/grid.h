#ifndef GRID_H
#define GRID_H

/*
 * Grid module
 *
 * Handles the simulation grid, including memory allocation,
 * initialization, reset, and user-defined heat sources.
 */

/* Simulation dimensions and rendering scale. */
extern int sim_width;
extern int sim_height;
extern int window_scale;

/*
 * Simulation data:
 *  - grid: current temperature values
 *  - next_grid: next state after simulation step
 *  - source_map: marks permanent heat sources
 */
extern float *grid;
extern float *next_grid;
extern unsigned char *source_map;

/* Initializes the grid with the given size and scale. */
void init_grid(int width, int height, int scale);

/* Resets the grid to the initial state (clears and adds center heat source). */
void reset_grid(void);

/* Frees all allocated memory related to the grid. */
void free_grid(void);

/*
 * Adds a permanent heat source at the given position.
 * The source is created within the specified radius.
 */
void add_heat_source(int mouse_x, int mouse_y, int radius);

#endif