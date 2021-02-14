#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (8192*8192)
#define block_Size 256 


__global__ void reduce(int *g_idata, unsigned long long int *g_odata)
{
    // Create a temporary vector accessable by all threads in the block    
    __shared__ unsigned long long int temp[block_Size];  
    // Get our global Array ID
    unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
    // Do work
    temp[threadIdx.x] = (i<N) ? (unsigned long long int)(g_idata[i]):0;
    if(i+blockDim.x < N)
       temp[threadIdx.x] += (unsigned long long int)(g_idata[i+blockDim.x]);
    // Thread barrier
    __syncthreads();

   // Reduce the shared memory to thread 0 on each block
   for(unsigned int s=blockDim.x/2; s>32; s>>=1) {
      if(threadIdx.x< s)
         temp[threadIdx.x] += temp[threadIdx.x+s];
      __syncthreads();
   }
   // fully unroll reduction within a single warp
/*   if ((block_Size >=  64) && (threadIdx.x < 32))
      temp[threadIdx.x] += temp[threadIdx.x+32];
   __syncthreads();

   if ((block_Size >=  32) && (threadIdx.x < 16))
      temp[threadIdx.x] += temp[threadIdx.x+16];
   __syncthreads();

   if ((block_Size >=  16) && (threadIdx.x <  8))
      temp[threadIdx.x] += temp[threadIdx.x+8];
   __syncthreads();

   if ((block_Size >=   8) && (threadIdx.x <  4))
      temp[threadIdx.x] += temp[threadIdx.x+4];
   __syncthreads();

   if ((block_Size >=   4) && (threadIdx.x <  2))
      temp[threadIdx.x] += temp[threadIdx.x+2];;
   __syncthreads();

   if ((block_Size >=   2) && (threadIdx.x <  1))
      temp[threadIdx.x] += temp[threadIdx.x+1];
   __syncthreads(); */
   if(threadIdx.x < 32)
   {
      temp[threadIdx.x] += temp[threadIdx.x+32];
      __syncthreads();
      temp[threadIdx.x] += temp[threadIdx.x+16];
      __syncthreads();
      temp[threadIdx.x] += temp[threadIdx.x+8];
      __syncthreads();
      temp[threadIdx.x] += temp[threadIdx.x+4];
      __syncthreads();
      temp[threadIdx.x] += temp[threadIdx.x+2];
      __syncthreads();
      temp[threadIdx.x] += temp[threadIdx.x+1];
      __syncthreads();
   }
   // Let thread 0 in all blocks add up the global sum
   if( threadIdx.x == 0)
      atomicAdd(g_odata, temp[threadIdx.x]);
}
 
int main( int argc, char* argv[] )
{
    double   t0, t1;    
    // Host vectors
    int *h_a;
    unsigned long long int *h_c;
    // Device vectors
    int *d_a;
    unsigned long long int *d_c;
 
    // Size, in bytes, of each vector
    int size = N*sizeof(int);

     // Allocate memory for each vector on host
    h_a = (int*)malloc(size);
    // Ok, so this is the global sum, a vector of length 1!
    h_c = (unsigned long long int*)malloc(sizeof(unsigned long long int));

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, sizeof(unsigned long long int));
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < N; i++ )
        h_a[i] = i;
    *h_c = 0;

    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_c, h_c, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
 
    // Execute the kernel
    int grid_Size = (int)ceil(((float)N/block_Size)/2);
    t0 = omp_get_wtime();
    reduce<<<grid_Size, block_Size>>>(d_a, d_c);
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
    t1 = omp_get_wtime();

    // Copy array back to host
    cudaMemcpy( h_c, d_c, sizeof(unsigned long long int), cudaMemcpyDeviceToHost );
    
    // Display the result
    printf("Sum = %llu\n", *h_c);
    printf("N*(N-1)/2 = %llu\n", (unsigned long long int) N*(N-1)/2);
    printf("Runtime: %.3f ms\n", (t1-t0)*1000);
    
    // Release device memory
    cudaFree(d_a); cudaFree(d_c);
    // Release host memory
    free(h_a); free(h_c);
 
    return 0;
}
