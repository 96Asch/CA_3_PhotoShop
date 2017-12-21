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
 const int index = blockIdx.y * blockDim.y + threadIdx.y;
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
   vec_mults(temp, mult, 4);
   color.x = temp[0];
   color.y = temp[1];
   color.z = temp[2];
   color.w = temp[3];
   RGBA_pack(*image_get_pixel_data(src, rowstride, xx, yy), color);      
}

__device__ void
vec_add(float *v1, float *v2, const int size)
{
 const int index = blockIdx.y * blockDim.y + threadIdx.y;
 if(index >= size)
    return;
 v1[index] += v2[index];
}

__global__ void
filters_cuda_rectangle(uint32_t *src, const int rowstride, const int x, const int y, const int width, const int height, const int rect_x, const int rect_y, const int rect_width, const int rect_height)
{
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;

  if (xx < x || xx >= width || yy < y || y >= height)
    return;

  const int start_x = rect_x < x ? x : rect_x;
  const int end_x = rect_x + rect_width < x + width ?
      rect_x + rect_width : x + width;

  const int start_y = rect_y < y ? y : rect_y;
  const int end_y = rect_y + rect_height < y + height ?
      rect_y + rect_height : y + height;

  if (xx >= start_x && yy >= start_y && xx <= end_x && yy <= end_y) {
     float rect_color[4], color_new[4];
     float alpha;
     rect_color[0] = 66/255.;
     rect_color[1] = 95/255.;
     rect_color[2] = 244/255.;
     rect_color[3] = 0.5;

     rgba_t color;
     RGBA_unpack(color, *image_get_pixel_data(src, rowstride, xx, yy));
     color_new[0] = color.x;
     color_new[1] = color.y;
     color_new[2] = color.z;
     color_new[3] = color.w;
     alpha = color.w;
 
     vec_mults(color_new, alpha, 4);
     vec_mults(rect_color, 1.0-alpha, 4);
     vec_add(color_new, rect_color, 4);
     color_new[3] = 0.5;
  
     color.x = color_new[0];
     color.y = color_new[1];
     color.z = color_new[2];
     color.w = color_new[3];
    RGBA_pack(*image_get_pixel_data(src, rowstride, xx, yy), color);
    }
}

__device__ static int
compute_hue (rgba_t color)
{
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

  return (int)hue;
}

__global__ void
filters_cuda_selective_greyscale(uint32_t *src, const int rowstride, const int x, const int y, const int width, const int height, const int hue, const int spread)
{
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;

  if (xx < x || xx >= width || yy < y || y >= height)
     return;

     float gray[4], color_new[4];

     rgba_t color;
     RGBA_unpack(color, *image_get_pixel_data(src, rowstride, xx, yy));
     color_new[0] = color.x;
     color_new[1] = color.y;
     color_new[2] = color.z;
     color_new[3] = color.w;
     
     float intensity =  color.w * ( 0.2126f * color.x + 0.7152f * color.y +
                          0.0722 * color.z);
     int diff = (int)abs(hue - compute_hue(color));
     float weight = (diff <= spread) ? (float)diff / (float)spread : 1.f;
    
     for(int i = 0; i < 3; i++)
     	 gray[i] = intensity;
     gray[3] = 1.f;

     vec_mults(gray, weight, 4);
     vec_mults(color_new, 1.f-weight, 4);
     vec_add(color_new, gray, 4);
     
     color.x = color_new[0];
     color.y = color_new[1];
     color.z = color_new[2];
     color.w = color_new[3];
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
  cudaMalloc(&result_D, sizeof(uint32_t) * (image->width * image->height));
  cudaMalloc(&image_D, sizeof(uint32_t) * (image->height * image->width));
  cudaMemcpy(image_D, image->data, sizeof(uint32_t) * (image->width * image->height), cudaMemcpyHostToDevice);
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
      filters_cuda_apply_gamma<<<numblocks, blocksize>>>(result_D, image->rowstride, 0, 0, image->width, image->height);
      filters_cuda_brightness<<<numblocks, blocksize>>>(result_D, image->rowstride, 0, 0, image->width, image->height,0.8, temp);
      filters_cuda_rectangle<<<numblocks, blocksize>>>(result_D, image->rowstride, 0, 0, image->width, image->height, 50, 50, result->width - 100, result->height - 100);
      filters_cuda_selective_greyscale<<<numblocks, blocksize>>>(result_D, image->rowstride, 0, 0, image->width, image->height, 40, 30);
      CUDA_ASSERT( cudaGetLastError() );


    }

  /* Pipeline is finished */
  CUDA_ASSERT(cudaEventRecord(stop));

  cudaMemcpy(result->data, result_D, sizeof(uint32_t) * (result->height * result->width), cudaMemcpyDeviceToHost);
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
