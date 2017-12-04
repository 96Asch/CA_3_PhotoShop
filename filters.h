/*
 * Skeleton code for use with Computer Architecture assignment 3, LIACS,
 * Leiden University.
 */

#ifndef __FILTERS_H__
#define __FILTERS_H__

#include "image.h"

/* The prototypes for the different filters that are implemented.
 * "filters-base.c" contains implementations of these filters that serve
 * as starting point. You will be created copies of this file to implement
 * optimized filters.
 *
 * All filters operate on the area indicated by
 * [x, y] - [x + width, y + height]. The individual filter functions do not
 * verify whether this area is within the bounds of the image. This is
 * left to the caller of the function!
 */

/* Copy image data from "src" to "dst". Images must be same dimensions. */
void filters_copy                       (image_t       *dst,
                                         const image_t *src,
                                         const int      x,
                                         const int      y,
                                         const int      width,
                                         const int      height);

/* Apply a fixed gamma correction of 2.2 to "image". */
void filters_apply_gamma                (image_t       *image,
                                         const int      x,
                                         const int      y,
                                         const int      width,
                                         const int      height);

/* Applies image brightness with given multiplication factor "mult"
 * to "image".
 */
void filters_brightness                 (image_t       *image,
                                         const int      x,
                                         const int      y,
                                         const int      width,
                                         const int      height,
                                         const float    mult);

/* Draws a rectangle atop "image". The rectangle is specified by
 * [rect_x, rect_y] - [rect_x + rect_width, rect_y + rect_height].
 * Note: only the area of [x, y] - [x + width, y + height] is processed!
 *
 * The "ATOP" operator is used, such that the rectangle is only drawn
 * is overlapping areas. The original image is left mostly intact.
 */
void filters_rectangle                  (image_t       *image,
                                         const int      x,
                                         const int      y,
                                         const int      width,
                                         const int      height,
                                         const int      rect_x,
                                         const int      rect_y,
                                         const int      rect_width,
                                         const int      rect_height);

/* Performs selective grayscale filter.
 * `hue` specifies the tint range to retain in degrees on the color circle
 * `spread` specifies the latitude around the given hue.
 */
void filters_selective_grayscale        (image_t       *image,
                                         const int      x,
                                         const int      y,
                                         const int      width,
                                         const int      height,
                                         const int      hue,
                                         const int      spread);

/* (Extra bonus opdracht, alleen voor de liefhebbers).
 * Performs gaussian blur filter `sigma` specifies the sigma parameters
 * in the gaussian function. The radius that is used for the blur is
 * (rougly) floor(3*sigma).
 */
void filters_gaussian_blur              (image_t       *dst,
                                         const image_t *src,
                                         const float    sigma);

#endif /* ! __FILTERS_H__ */
