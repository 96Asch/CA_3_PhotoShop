/*
 * Skeleton code for use with Computer Architecture assignment 3, LIACS,
 * Leiden University.
 */

#ifndef __PIPELINE_H__
#define __PIPELINE_H__

#include "image.h"

/* Generic function to run an image pipeline. "image" is the source
 * image that is left unmodified. Result to be written to "result".
 * The pipeline is repeated "n_repeat" times.
 */
void run_filters(image_t *result, const image_t *image, const int n_repeat);

#endif /* __PIPELINE_H__ */
