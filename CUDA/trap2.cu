#include <stdio.h>
#include <stdlib.h>

#define N 4096
#define block_Size 256

/* function to integrate, defined as a function on the GPU device */
__device__ float myfunction(float a)
{
   return a*a+2.0*a + 3.0;
}
 
/* kernel function to compute the summation used in the trapezoidal rule 
   for numerical integration */
__global__ void integratorKernel(float *a, float start, float deltaX)
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;

   if (id<N)
   {
      float x = start + (float)id * deltaX;
      atomicAdd(a, myfunction(x)+myfunction(x+deltaX));
   }
}
 
int main( int argc, char* argv[] )
{
   float end = 1.0, start = 0.0;
   // deltaX
   float deltaX = (end-start)/(float) N;

   // error code variable
   cudaError_t errorcode = cudaSuccess;

   // Allocate array on host and device
   float *sum_h;
   sum_h = (float*)malloc(sizeof(float));
   *sum_h = 0.0;

   float *sum_d;
   if (( errorcode = cudaMalloc((void **)&sum_d, sizeof(float)))!= cudaSuccess)
   {
      printf("cudaMalloc(): %s/n", cudaGetErrorString(errorcode));
      exit(1);
   }

   // Copy values from host to device
   if((errorcode = cudaMemcpy( sum_d, sum_h, sizeof(float), cudaMemcpyHostToDevice))
      !=cudaSuccess)
   {
      printf("cudaMemcpy(): %s\n", cudaGetErrorString(errorcode));
      exit(1);
   }
   
   // Do the integration
   int grid_Size = N/block_Size + ( N % block_Size == 0 ? 0:1);
   integratorKernel <<< grid_Size, block_Size >>> (sum_d, start, deltaX);

   // Copy results from device to host
   if((errorcode = cudaMemcpy( sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost))
      !=cudaSuccess)
   {
      printf("cudaMemcpy(): %s\n", cudaGetErrorString(errorcode));
      exit(1);
   }

   printf("The integral is: %f\n", (*sum_h)*deltaX/2.0);

   // clean up
   free(sum_h);
   cudaFree(sum_d);

   return 0;
}
