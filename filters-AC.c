/*
 * Skeleton code for use with Computer Architecture assignment 3, LIACS,
 * Leiden University.
 */

#include "filters.h"

#include "gamma.h"

#include <math.h>
#include <stdlib.h>
#include <assert.h>

/* See filters.h for a brief description of these functions. */

void filters_copy(image_t *dst, const image_t *src, const int x, const int y,
		const int width, const int height) {
	const int end = (x + width) * (y + height);
	for (int xx = x; xx < end; xx++) {
		dst->data[xx] = src->data[xx];
	}

}

void filters_apply_gamma(image_t *image, const int x, const int y,
		const int width, const int height) {
	const int end = (x + width) * (y + height);
	for (int xx = x; xx < end; xx++) {
		/* Get the i-th pixel */
		uint32_t pixel = image->data[xx];

		/* Copy alpha value */
		uint32_t result = pixel & 0xFF;

		/* Gamma correct each color component using bit shifts (see gamma.h) */
		result |= GAMMA[0xFF & (pixel >> 24)] << 24; // red
		result |= GAMMA[0xFF & (pixel >> 16)] << 16; // green
		result |= GAMMA[0xFF & (pixel >> 8)] << 8;   // blue

		/* Store the resulting pixel in the output buffer */
		image->data[xx] = result;
	}
}

void filters_brightness(image_t *image, const int x, const int y,
		const int width, const int height, const float mult) {
	const int end = (x + width) * (y + height);
	for (int xx = x; xx < end; xx++) {
		/* Unpack pixel, multiply with scalar "mult" and pack the pixel
		 * again.
		 */
		rgba_t color;
		RGBA_unpack(color, image->data[xx]);
		RGBA_mults(color, color, mult);
		RGBA_pack(image->data[xx], color);
	}
}

void
filters_rectangle(image_t *image,
                  const int x, const int y,
                  const int width, const int height,
                  const int rect_x, const int rect_y,
                  const int rect_width, const int rect_height)
{
  /* Determine in which ranges we need to draw the rectangle */
  const int start_x = rect_x < x ? x : rect_x;
  const int end_x = rect_x + rect_width < x + width ?
      rect_x + rect_width : x + width;

  const int start_y = rect_y < y ? y : rect_y;
  const int end_y = rect_y + rect_height < y + height ?
      rect_y + rect_height : y + height;

  /* Visit all pixels where the rectangle needs to be drawn and apply
   * the ATOP operator.
   */
  int offset = start_y * height;
  for (int yy = start_y; yy < end_y; yy++)
    {
      for (int xx = start_x; xx < end_x; xx++)
        {
          rgba_t color, rect_color;
          float color_alpha;
          int index = offset+xx;
          /* Hard-coded blue color for the rectangle. Alpha value of 0.5 */
          RGBA(rect_color, 66/255., 95/255., 244/255., 0.5);

          /* Perform ATOP operator. */
          RGBA_unpack(color, image->data[index]);
          color_alpha = color.w;
          RGBA_mults(color, color, color_alpha);
          RGBA_mults(rect_color, rect_color, 1.0 - color_alpha);
          RGBA_add(color, color, rect_color);
          color.w = 0.5;
          RGBA_pack(image->data[offset+xx], color);
        }
      offset += width;
    }
}
/* Computes the hue (tint) of an RGB(A) quadruple.
 *
 * See http://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/
 */
static int compute_hue(rgba_t color) {
	float hue = 0.f;

	/* Find the 'largest' component of either of r, g or b */
	float c_max = fmax(fmax(color.x, color.y), color.z);
	float c_min = fmin(fmin(color.x, color.y), color.z);

	/* The hue depends on which component in the largest */
	if (c_max == color.x)
		hue = (color.y - color.z) / (c_max - c_min);
	else if (c_max == color.y)
		hue = 2.f + (color.z - color.x) / (c_max - c_min);
	else
		hue = 4.f + (color.x - color.y) / (c_max - c_min);

	/* Ensure hue is in the range of [0-360]. */
	hue *= 60.f;
	if (hue < 0)
		hue += 360.f;

	return (int) hue;
}

void filters_selective_grayscale(image_t *image, const int x, const int y,
		const int width, const int height, const int hue, const int spread) {
	/* We compute the 'selective grayscale' here, this is the general idea:
	 * - Iterate over every pixel in the source image.
	 * - We compute each pixel's 'intensity' from its RGB triplet.
	 * - We compute the hue of each pixel.
	 * - When the hue differs no more than 'spread' from the given hue,
	 * we retain the original color, otherwise we use the computed gray value.
	 * - To reduce artifacts, we 'weigh' the color and gray components,
	 * based on the difference between the pixel's hue and the given hue.
	 */

	const int end = (x + width) * (y + height);
	for (int xx = x; xx < end; xx++) {
		rgba_t color;
		RGBA_unpack(color, image->data[xx]);

		/* We use CIE 1931 weights multiplied by alpha, to compute
		 * the 'intensity'.
		 * Y = A( 0.2126R + 0.7152G + 0.0722B )
		 *   (remember that r,g,b,a=x,y,z,w)
		 */
		float intensity = color.w
				* (0.2126f * color.x + 0.7152f * color.y + 0.0722 * color.z);

		/* Compute the 'hue' (tint) and the difference with the
		 * given hue...
		 */
		int diff = (int) abs(hue - compute_hue(color));

		/* ...this difference determines whether we pick the gray or the
		 * original color. We use a linear weight to reduce artifacts in
		 * the final image.
		 */
		float weight = (diff <= spread) ? (float) diff / (float) spread : 1.f;

		/* Apply the weights to the 'color' and 'gray' components. */
		rgba_t gray;
		RGBA(gray, intensity, intensity, intensity, 1.f);

		RGBA_mults(gray, gray, weight);
		RGBA_mults(color, color, 1.f - weight);

		/* Finally, add both components to produce the resulting pixel. */
		RGBA_add(color, color, gray);
		RGBA_pack(image->data[xx], color);

	}
}

/* Deze functie wordt alleen aangepast voor de bonus opgave. */

void filters_gaussian_blur(image_t *dst, const image_t *src, const float sigma) {
	/* This function computes a 'gaussian blur'. Gaussian blur is basically
	 * defined as 'applying a gaussian kernel' to each pixel, using its
	 * neighbours as input. This 'kernel' is a matrix of coefficents from
	 * the gaussian function: exp( i^2 / 2 * sigma^2 ) Here 'i' is the
	 * distance from the original pixel and 'sigma' is the amount of blur.
	 *
	 * We use a very simplified and naive version here, consisting of the
	 * following steps:
	 * - We make two passes over the image: one horizontal and one vertical
	 *   The intermediate result is stored in the 'dst' buffer.
	 * - In each pass, we sample the neighbours of each pixel to compute the blur.
	 * - The number of neighbours we sample is the 'kernel size', defined as
	 *   1+2*floor(sigma*3).
	 * - We accumulate all the neighbours weighted by the gaussian coefficient
	 *   (see above).
	 * - We normalize the accumulated values and write it to the buffer.
	 *
	 * See also: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch40.html
	 */

	assert(image_equal_dimensions(dst, src) == true);

	/* Number of samples before and after the center pixel */
	const int spread = (int) (sigma * 3.f);

	/* Normalization constant */
	const float norm = 1.f / (sqrt(2 * M_PI) * sigma);
	const int direction[2][2] = { { 0, 1 }, { 1, 0 } };

	/* Weights for x and y to determine the direction (horizontal or vertical) */
	int dir[2];
	const image_t *src_img = src;

	for (int d = 0; d < 2; d++) {
		/* Set the direction to either horizontal (d=1) or vertical (d=0) */
		dir[0] = direction[d][0];
		dir[1] = direction[d][1];

		for (int x = 0; x < src->width; x++) {
			for (int y = 0; y < src->height; y++) {
				/* Obtain the center pixel from the array */
				rgba_t color;
				RGBA_unpack(color, *image_get_pixel(src_img, x, y));

				for (int i = 1; i <= spread; i++) {
					/* Compute the gaussian coefficient for i, the naive way
					 * On the GPU, there are many ways to optimize this,
					 * for example by precalculating it in shared memory
					 */
					float coeff = exp(
							-.5f * (float) i * (float) i / (sigma * sigma));

					/* Accumulate the neighbouring pixels:
					 *   color += ith_prev_neighbour * coeff
					 *   color += ith_next_neighbour * coeff
					 */

					rgba_t temp;
					int clamped_x, clamped_y;

					/* Left or up */
					clamped_x = CLAMP(x - dir[0] * i, 0, src->width - 1);
					clamped_y = CLAMP(y - dir[1] * i, 0, src->height - 1);
					RGBA_unpack(temp,
							*image_get_pixel(src_img, clamped_x, clamped_y));
					RGBA_mults(temp, temp, coeff);
					RGBA_add(color, color, temp);

					/* Right or below */
					clamped_x = CLAMP(x + dir[0] * i, 0, src->width - 1);
					clamped_y = CLAMP(y + dir[1] * i, 0, src->height - 1);
					RGBA_unpack(temp,
							*image_get_pixel(src_img, clamped_x, clamped_y));
					RGBA_mults(temp, temp, coeff);
					RGBA_add(color, color, temp);
				}

				/* Because of all the additions, the values are not in range
				 * [0-1]. So 'normalize' and finally write to the
				 * destination buffer.
				 */
				RGBA_mults(color, color, norm);
				RGBA_pack(*image_get_pixel(dst, x, y), color);
			}
		}

		/* Change the source image for the second pass */
		src_img = dst;
	}
}
