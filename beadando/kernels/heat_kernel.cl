/*
 * OpenCL kernel for one heat diffusion step.
 *
 * Each work-item computes the new temperature of one grid cell
 * based on its four direct neighbors.
 *
 * Boundary cells are kept at zero temperature.
 * Cells marked in source_map remain constant heat sources.
 */
__kernel void heat_step(__global const float *current,
                        __global float *next,
                        __global const uchar *source_map,
                        const int width,
                        const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    /* Ignore padded work-items outside the valid simulation domain. */
    if (x >= width || y >= height)
    {
        return;
    }

    int idx = y * width + x;

    /* Keep boundary cells cold. */
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
    {
        next[idx] = 0.0f;
        return;
    }

    /* Preserve permanent heat sources. */
    if (source_map[idx])
    {
        next[idx] = 1.0f;
        return;
    }

    /* Read the four direct neighbors. */
    float up = current[(y - 1) * width + x];
    float down = current[(y + 1) * width + x];
    float left = current[y * width + (x - 1)];
    float right = current[y * width + (x + 1)];

    /* Compute the new temperature as the average of neighbors. */
    next[idx] = 0.25f * (up + down + left + right);
}