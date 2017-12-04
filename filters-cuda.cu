/*
 * Skeleton code for use with Computer Architecture assignment 3, LIACS,
 * Leiden University.
 */

#include "filters.h"

#include "gamma.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Some simple assert macro's to handle CUDA-errors more easily
#define CUDA_ASSERT(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void
cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
              file, line);

      if (abort)
        exit(code);
    }
}


/*
 * TODO:
 * Put your kernels and device functions here.  Remember that every kernel
 * function should be declared as `__global__ void`
 * Every function you want to call from a kernel, should begin with
 * `__device__`
 */

__global__ void
filters_cuda_copy(uint32_t *dst, const uint32_t *src, const int rowstride,
                  const int x, const int y,
                  const int width, const int height)
{
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;

  if (xx < x || xx >= width || yy < y || y >= height)
    return;

  /* Get the pixel in src and store in dst. */
  uint32_t pixel = *image_get_pixel_data(src, rowstride, xx, yy);
  *image_get_pixel_data(dst, rowstride, xx, yy) = pixel;
}


/* To keep things easy we put run_filters() for the GPU version here,
 * instead of in a separate "pipeline.c" file.
 */
extern "C"
{

__host__ void
run_filters(image_t *result, const image_t *image, const int n_repeat)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Calculate the block size and the number of thread blocks */
  const dim3 blocksize(16, 16);
  const dim3 numblocks((image->width % blocksize.x) ?
                          image->width / blocksize.x + 1 :
                          image->width / blocksize.x,
                       (image->height % blocksize.y) ?
                          image->height / blocksize.y + 1 :
                          image->height / blocksize.y);

  /* TODO: Allocate buffers to contain initial and result images */

  /* Copy the input image to the initial buffer */

  /* Start the timer */
  CUDA_ASSERT(cudaEventRecord(start));

  for(int i = 0; i < n_repeat; i++)
    {
      /* Copy the initial buffer to the result buffer. This is similar
       * to the operation done in the CPU code to allow a fair comparison.
       */
/* Example CUDA kernel call:
      filters_cuda_copy<<<numblocks, blocksize>>>(result_D, image_D,
                                                  image->rowstride,
                                                  0, 0,
                                                  image->width,
                                                  image->height);
      CUDA_ASSERT( cudaGetLastError() );
*/

    }

  /* Pipeline is finished */
  CUDA_ASSERT(cudaEventRecord(stop));

  /* TODO: Copy the result buffer back to result->data */

  /* Synchronize and print statistics */
  CUDA_ASSERT(cudaEventSynchronize(stop ));
  float msec = 0;
  CUDA_ASSERT(cudaEventElapsedTime(&msec, start, stop));

  fprintf(stderr, "elapsed time GPU: %f s\n", msec / 1000.);
}

} /* end extern "C" */
