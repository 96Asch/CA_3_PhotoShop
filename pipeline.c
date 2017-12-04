/*
 * Skeleton code for use with Computer Architecture assignment 3, LIACS,
 * Leiden University.
 */

#include "pipeline.h"
#include "filters.h"

#include "timing.h"

/* The image pipeline to experiment with. It carries out the following stes:
 * - Make a copy of "image" in "result". Subsequent filters are applied on
 *   "result".
 * - Apply a fixed gamma correction of 2.2.
 * - Apply image brightness.
 * - Perform selective grayscale.
 * - Draw a translucent rectangle on top, inset 50 pixels from the border
 *   of the image.
 */

void
run_filters(image_t *result, const image_t *image, const int n_repeat)
{
  /* Important: get_time() requests "CPU time" which includes
   * the CPU time consumed by all threads in the process. So for
   * a multi-threaded code this might not give the timing information
   * you want. You can change get_time() in timing.h to use for example
   * CLOCK_REALTIME (which is potentially less accurate) instead
   * of CLOCK_PROCESS_CPUTIME_ID.
   */
  struct timespec start_time, end_time;
  get_time(&start_time);

  for (int Z = 0; Z < n_repeat; Z++)
    {
      filters_copy(result, image, 0, 0, image->width, image->height);

      filters_apply_gamma(result, 0, 0, result->width, result->height);

      filters_brightness(result, 0, 0, result->width, result->height, 0.8);

      filters_selective_grayscale(result,
                                  0, 0, result->width, result->height, 40, 30);

      filters_rectangle(result,
                        0, 0, result->width, result->height,
                        50, 50, result->width - 100, result->height - 100);

#if 0
      /* Voor de liefhebbers. */
      filters_gaussian_blur(result, temp2, 3);
#endif
    }

  get_time(&end_time);

  print_elapsed_time("filters", &end_time, &start_time);
}
