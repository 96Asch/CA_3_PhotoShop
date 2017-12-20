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

  int blockSize = 128;

  /* Determine in which ranges we need to draw the rectangle */
  const int xx = 0, yy = 0, rect_x = 50, rect_y = 50, rect_height = result->height - 100,
		  rect_width = result->width - 100;
  const int start_x = rect_x < xx ? xx : rect_x;
  const int end_x = rect_x + rect_width < xx + result->width ?
      rect_x + rect_width : xx + result->width;

  const int start_y = rect_y < yy ? yy : rect_y;
  const int end_y = rect_y + rect_height < yy + result->height ?
      rect_y + rect_height : yy + result->height;

  for (int Z = 0; Z < n_repeat; Z++)
    {
	  for (int x = 0; x < image->height; x++) {
	      for (int y = 0; y < image->width; y++) {
				  filters_copy(result, image, y, x, image->width, image->height);
				  filters_apply_gamma(result, y, x, result->width, result->height);

				  filters_brightness(result, y, x, result->width, result->height, 0.8);

				  filters_selective_grayscale(result,
											  y, x, result->width, result->height, 40, 30);

			      filters_rectangle(result,
			                        y, x, result->width, result->height,
									start_x, start_y, end_x, end_y
			                       );

			#if 0
				  /* Voor de liefhebbers. */
				  filters_gaussian_blur(result, temp2, 3);
			#endif
	      }
	  }
    }
  get_time(&end_time);

  print_elapsed_time("filters", &end_time, &start_time);
}
