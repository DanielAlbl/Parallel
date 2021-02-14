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
__global__ void integratorKernel(float *a, float c, float deltaX)
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   float x = c + (float)id * deltaX;

   if (id<N)
      a[id] = myfunction(x)+myfunction(x+deltaX);
}
 
int main( int argc, char* argv[] )
{
   double end = 1.0, start = 0.0;

   // deltaX
   float deltaX = (end-start)/(double) N;

   // error code variable
   cudaError_t errorcode = cudaSuccess;

   // Size of the arrays in bytes
   int size = N*sizeof(float);

   // Allocate array on host and device
   float* a_h = (float *)malloc(size);

   float* a_d;  
   if (( errorcode = cudaMalloc((void **)&a_d,size))!= cudaSuccess)
   {
      printf("cudaMalloc(): %s/n", cudaGetErrorString(errorcode));
      exit(1);
   }

   // Do calculation on device
   int grid_Size = N/block_Size + ( N % block_Size == 0 ? 0:1);
   printf("blocks: %d\n", grid_Size);
   printf("block size: %d\n ", block_Size);

   integratorKernel <<< grid_Size, block_Size >>> (a_d, start, deltaX);

   // Copy results from device to host
   if((errorcode = cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost))
      !=cudaSuccess)
   {
      printf("cudaMemcpy(): %s\n", cudaGetErrorString(errorcode));
      exit(1);
   }


   // Add up results
   float sum = 0.0;
   for(int i=0; i<N; i++) 
      sum += a_h[i];
   sum *= deltaX/2.0;

   printf("The integral is: %f\n", sum);

   // clean up
   free(a_h);
   cudaFree(a_d);

   return 0;
}
