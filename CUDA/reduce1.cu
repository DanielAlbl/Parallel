#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N (8192*8192)
#define block_Size 256 


__global__ void reduce(int *g_idata, unsigned long long int *g_odata)
{
    // Create a temporary vector accessable by all threads in the block    
    __shared__ unsigned long long int temp[block_Size];  
    // Get our global Array ID
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    // Do work
    temp[threadIdx.x] = (unsigned long long int) g_idata[i];
    // Thread barrier
    __syncthreads();

   // Reduce the shared memory to thread 0 on each block
   for(unsigned int s=1; s<blockDim.x; s*=2) {
      if(threadIdx.x%(2*s) == 0)
         temp[threadIdx.x] += temp[threadIdx.x+s];
      __syncthreads();
   }

   // Let thread 0 in all blocks add up the global sum
   if(threadIdx.x == 0)
      atomicAdd(g_odata, temp[threadIdx.x]);
}
 
int main( int argc, char* argv[] )
{
    struct timeval begin, end;
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
    int grid_Size = (int)ceil((float)N/block_Size);
    gettimeofday(&begin, NULL);
    reduce<<<grid_Size, block_Size>>>(d_a, d_c);
    // Wait for the GPU to finish
    // cudaDeviceSynchronize();   
    gettimeofday(&end, NULL);
    if (end.tv_usec < begin.tv_usec) {
        end.tv_usec += 1000000;
        begin.tv_sec += 1;
    }
    
    // Copy array back to host
    cudaMemcpy( h_c, d_c, sizeof(unsigned long long int), cudaMemcpyDeviceToHost );
    
    // Display the result
    printf("Sum = %llu\n", *h_c);
    printf("N*(N-1)/2 = %llu\n", (unsigned long long int) N*(N-1)/2);
    printf("Runtime: %.3lf ms\n", (double) ((end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)/1000.0));
 
    // Release device memory
    cudaFree(d_a); cudaFree(d_c);
    // Release host memory
    free(h_a); free(h_c);
 
    return 0;
}
