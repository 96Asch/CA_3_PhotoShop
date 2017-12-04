/*
 * Skeleton code for use with Computer Architecture assignment 3, LIACS,
 * Leiden University.
 */

#include "pipeline.h"
#include "timing.h"

#include <stdio.h>
#include <stdlib.h>

#define N_REPEAT 5

int
main(int argc, char **argv)
{
  if (argc < 2)
    {
      fprintf(stderr, "usage: %s <infile> [outfile]\n", argv[0]);
      fprintf(stderr, "\n  where <infile> and [outfile] are PNG files.\n");
      fprintf(stderr, "\n  [outfile] is an optional parameter.\n");
      return -1;
    }

  /* Load PNG image */
  struct timespec start_time, end_time;
  get_time(&start_time);

  image_t *image = image_new_from_pngfile(argv[1]);
  if (!image)
    return -1;

  image_t *result = image_new_from_image(image);

  get_time(&end_time);
  print_elapsed_time("file load", &end_time, &start_time);

  /* Run a filters pipeline, depending on which "pipeline.c" is linked
   * with the executable.
   */
  run_filters(result, image, N_REPEAT);

  /* Save result if desired */
  int retval = 0;
  if (argv[2] && !image_save_as_pngfile(result, argv[2]))
    retval = -1;

  image_free(image);
  image_free(result);

  return retval;
}
