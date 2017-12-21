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
filters_cuda_apply_gamma(uint32_t *src, const int rowstride, const int x, const int y, const int width, const int height)
{
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;
  
   if (xx < x || xx >= width || yy < y || y >= height)
    return;

  uint32_t pixel = *image_get_pixel_data(src, rowstride, xx, yy);
         /* Copy alpha value */
          uint32_t result = pixel & 0xFF;

          /* Gamma correct each color component using bit shifts (see gamma.h) */
          result |= GAMMA[0xFF & (pixel >> 24)] << 24; // red
          result |= GAMMA[0xFF & (pixel >> 16)] << 16; // green
          result |= GAMMA[0xFF & (pixel >> 8)] << 8;   // blue
  *image_get_pixel_data(src, rowstride, xx, yy) = result;
}

__device__ void
vec_mults(float* src, const float scalar, const int size) 
{
 const int index = blockIdx.x * blockDim.x + threadIdx.x;
 if(index >= size)
    return;

 src[index] *= scalar;
}

__global__ void
filters_cuda_brightness(uint32_t *src, const int rowstride, const int x, const int y, const int width, const int height, const float mult, float* temp)
{
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;

   if (xx < x || xx >= width || yy < y || y >= height)
    return;

   rgba_t color;
   RGBA_unpack(color, *image_get_pixel_data(src, rowstride, xx, yy));
   temp[0] = color.x;
   temp[1] = color.y;
   temp[2] = color.z;
   temp[3] = color.w;
   int size = 4;
   vec_mults(temp, mult, size);
   color.x = temp[0];
   color.y = temp[1];
   color.z = temp[2];
   color.w = temp[3];
   RGBA_pack(*image_get_pixel_data(src, rowstride, xx, yy), color);      
}

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

  uint32_t *result_D;
  uint32_t *image_D;
  cudaMalloc(&result_D, sizeof(uint32_t) * image->width);
  cudaMalloc(&image_D, sizeof(uint32_t) * image->width);
  cudaMemcpy(image_D, &image->data, sizeof(uint32_t) * image->width, cudaMemcpyHostToDevice);
  /* Start the timer */
  CUDA_ASSERT(cudaEventRecord(start));
  float *temp;
  cudaMalloc(&temp, sizeof(float) * 4);
  for(int i = 0; i < n_repeat; i++)
    {
      /* Copy the initial buffer to the result buffer. This is similar
       * to the operation done in the CPU code to allow a fair comparison.
       */

      filters_cuda_copy<<<numblocks, blocksize>>>(result_D, image_D,
                                                  image->rowstride,
                                                  0, 0,
                                                  image->width,
                                                  image->height);
//      filters_cuda_apply_gamma<<<numblocks, blocksize>>>(result_D, image->rowstride, 0, 0, image->width, image->height);
 //     filters_cuda_brightness<<<numblocks, blocksize>>>(result_D, image->rowstride, 0, 0, image->width, image->height,0.8, temp);
      CUDA_ASSERT( cudaGetLastError() );


    }

  /* Pipeline is finished */
  CUDA_ASSERT(cudaEventRecord(stop));

  cudaMemcpy(result->data, result_D, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(result_D);
  cudaFree(image_D);
  cudaFree(temp);
  /* Synchronize and print statistics */
  CUDA_ASSERT(cudaEventSynchronize(stop ));
  float msec = 0;
  CUDA_ASSERT(cudaEventElapsedTime(&msec, start, stop));

  fprintf(stderr, "elapsed time GPU: %f s\n", msec / 1000.);
}

} /* end extern "C" */
