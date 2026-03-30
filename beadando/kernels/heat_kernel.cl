/*
 * OpenCL kernel for one heat diffusion step.
 *
 * Each work-item is responsible for computing the next temperature
 * value of a single grid cell.
 *
 * The simulation uses:
 * - fixed cold boundary cells (temperature = 0.0)
 * - permanent heat source cells (temperature = 1.0)
 * - diffusion based on the average of the four direct neighbors
 */
__kernel void heat_step(__global float* current,
                        __global float* next,
                        __global uchar* source_map,
                        int width,
                        int height)
{
    /* Get the 2D coordinates of the current work-item. */
    int x = get_global_id(0);
    int y = get_global_id(1);

    /* Convert 2D coordinates to a 1D array index. */
    int idx = y * width + x;

    /*
     * Boundary condition:
     * the outer border of the grid is kept at zero temperature.
     */
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
    {
        next[idx] = 0.0f;
        return;
    }

    /*
     * Heat source condition:
     * cells marked in source_map remain at maximum temperature.
     */
    if (source_map[idx])
    {
        next[idx] = 1.0f;
        return;
    }

    /* Read the four direct neighboring temperature values. */
    float up = current[(y - 1) * width + x];
    float down = current[(y + 1) * width + x];
    float left = current[y * width + (x - 1)];
    float right = current[y * width + (x + 1)];

    /*
     * Compute the next temperature as the average
     * of the four neighboring cells.
     */
    next[idx] = 0.25f * (up + down + left + right);
}