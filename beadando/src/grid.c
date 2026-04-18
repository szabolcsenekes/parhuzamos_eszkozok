#include <stdio.h>
#include <stdlib.h>
#include "grid.h"

/*
 * Resets the simulation to its initial state.
 *
 * All cells are cleared to zero temperature, then a fixed
 * heat source is placed in the center of the grid.
 */
void reset_grid(Grid *g)
{
    int size = g->width * g->height;

    for (int i = 0; i < size; i++)
    {
        g->current[i] = 0.0f;
        g->next[i] = 0.0f;
        g->source_map[i] = 0;
    }

    int cx = g->width / 2;
    int cy = g->height / 2;

    for (int y = cy - 10; y <= cy + 10; y++)
    {
        for (int x = cx - 10; x <= cx + 10; x++)
        {
            if (x >= 0 && x < g->width && y >= 0 && y < g->height)
            {
                int idx = y * g->width + x;
                g->current[idx] = 1.0f;
                g->next[idx] = 1.0f;
                g->source_map[idx] = 1;
            }
        }
    }
}

/*
 * Allocates memory for the simulation grid and stores
 * the selected simulation parameters.
 */
void init_grid(Grid *g, int width, int height, int scale)
{
    g->width = width;
    g->height = height;
    g->window_scale = scale;

    g->current = (float *)malloc(sizeof(float) * width * height);
    g->next = (float *)malloc(sizeof(float) * width * height);
    g->source_map = (unsigned char *)malloc(sizeof(unsigned char) * width * height);

    if (g->current == NULL || g->next == NULL || g->source_map == NULL)
    {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    reset_grid(g);
}

/*
 * Releases all dynamically allocated memory used by the grid.
 */
void free_grid(Grid *g)
{
    free(g->current);
    free(g->next);
    free(g->source_map);

    g->current = NULL;
    g->next = NULL;
    g->source_map = NULL;
    g->width = 0;
    g->height = 0;
    g->window_scale = 1;
}

/*
 * Adds a new permanent heat source at the given position.
 */
void add_heat_source(Grid *g, int mouse_x, int mouse_y, int radius)
{
    if (mouse_x <= 0 || mouse_x >= g->width - 1 ||
        mouse_y <= 0 || mouse_y >= g->height - 1)
    {
        return;
    }

    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            int x = mouse_x + dx;
            int y = mouse_y + dy;

            if (x > 0 && x < g->width - 1 &&
                y > 0 && y < g->height - 1)
            {
                int idx = y * g->width + x;
                g->source_map[idx] = 1;
                g->current[idx] = 1.0f;
                g->next[idx] = 1.0f;
            }
        }
    }
}