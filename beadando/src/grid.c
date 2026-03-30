#include <stdio.h>
#include <stdlib.h>
#include "grid.h"

/* Current simulation dimensions and rendering scale. */
int sim_width = 0;
int sim_height = 0;
int window_scale = 1;

/*
 * grid        - current temperature field
 * next_grid   - next temperature field after one simulation step
 * source_map  - marks cells that act as permanent heat sources
 */
float *grid = NULL;
float *next_grid = NULL;
unsigned char *source_map = NULL;

/*
 * Resets the simulation to its initial state.
 *
 * All cells are cleared to zero temperature, then a fixed heat source
 * is placed in the center of the grid.
 */
void reset_grid(void)
{
    int size = sim_width * sim_height;

    /* Clear temperature fields and source map. */
    for (int i = 0; i < size; i++)
    {
        grid[i] = 0.0f;
        next_grid[i] = 0.0f;
        source_map[i] = 0;
    }

    /* Compute the center of the simulation grid. */
    int cx = sim_width / 2;
    int cy = sim_height / 2;

    /*
     * Create a square heat source in the center.
     * These cells remain permanently hot during the simulation.
     */
    for (int y = cy - 10; y <= cy + 10; y++)
    {
        for (int x = cx - 10; x <= cx + 10; x++)
        {
            if (x >= 0 && x < sim_width && y >= 0 && y < sim_height)
            {
                int idx = y * sim_width + x;
                grid[idx] = 1.0f;
                next_grid[idx] = 1.0f;
                source_map[idx] = 1;
            }
        }
    }
}

/*
 * Allocates memory for the simulation grid and stores the selected
 * simulation parameters.
 *
 * width  - grid width
 * height - grid height
 * scale  - rendering scale for the SDL window
 */
void init_grid(int width, int height, int scale)
{
    sim_width = width;
    sim_height = height;
    window_scale = scale;

    grid = (float *)malloc(sizeof(float) * sim_width * sim_height);
    next_grid = (float *)malloc(sizeof(float) * sim_width * sim_height);
    source_map = (unsigned char *)malloc(sizeof(unsigned char) * sim_width * sim_height);

    /* Stop the program if memory allocation fails. */
    if (grid == NULL || next_grid == NULL || source_map == NULL)
    {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    /* Initialize the newly allocated grid. */
    reset_grid();
}

/*
 * Releases all dynamically allocated memory used by the simulation.
 */
void free_grid(void)
{
    free(grid);
    free(next_grid);
    free(source_map);

    grid = NULL;
    next_grid = NULL;
    source_map = NULL;

    sim_width = 0;
    sim_height = 0;
}

/*
 * Adds a new permanent heat source at the given mouse position.
 *
 * The source is created as a square area with the given radius,
 * and it is stored both in the temperature fields and in source_map.
 */
void add_heat_source(int mouse_x, int mouse_y, int radius)
{
    /* Ignore clicks outside the valid simulation area. */
    if (mouse_x <= 0 || mouse_x >= sim_width - 1 ||
        mouse_y <= 0 || mouse_y >= sim_height - 1)
    {
        return;
    }

    /* Mark a square neighborhood around the clicked point as a heat source. */
    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            int x = mouse_x + dx;
            int y = mouse_y + dy;

            if (x > 0 && x < sim_width - 1 &&
                y > 0 && y < sim_height - 1)
            {
                int idx = y * sim_width + x;
                source_map[idx] = 1;
                grid[idx] = 1.0f;
                next_grid[idx] = 1.0f;
            }
        }
    }
}