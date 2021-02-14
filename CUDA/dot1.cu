#include <stdio.h>
#include <stdlib.h>

#define N (2048*2048)
#define block_Size 256 

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecDot1(int *a, int *b, unsigned long long int *c)
{
    __shared__ int temp[block_Size];
    
    // Get our global thread ID
    int index = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Create a temporary vector accessable by all threads in the block
    temp[threadIdx.x] = a[index] * b[index];
    // Thread barrier
    __syncthreads();

   // Let thread 0 in all blocks add up the temp vector
   if( threadIdx.x == 0) {
      unsigned long long int sum = 0;
      for( int i = 0; i<block_Size; i++)
         sum += (unsigned long long int)temp[i];
      // Add result to the global sum
      atomicAdd(c, sum);
   }
}
 
int main( int argc, char* argv[] )
{
    // Host vectors
    int *h_a, *h_b;
    unsigned long long int *h_c;
 
    // Device vectors
    int *d_a, *d_b;
    unsigned long long int *d_c;
 
    // Size, in bytes, of each vector
    int size = N*sizeof(int);
 
    // Allocate memory for each vector on host
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    // Ok, so this is the global sum, a vector of length 1!
    h_c = (unsigned long long int*)malloc(sizeof(unsigned long long int));
 
    // Allocate memory for each vector on GPU
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, sizeof(unsigned long long int));
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < N; i++ ) {
        h_a[i] = i;
        h_b[i] = 1;
    }
    *h_c = 0;
 
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_c, h_c, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
 
    // Execute the kernel
    int grid_Size = N/block_Size + ( N % block_Size == 0 ? 0:1);
    vecDot1<<<grid_Size, block_Size>>>(d_a, d_b, d_c);
 
    // Copy array back to host
    cudaMemcpy( h_c, d_c, sizeof(unsigned long long int), cudaMemcpyDeviceToHost );
    
    // Display the result
    printf("A.B = %llu\n", *h_c);
    printf("N*(N-1)/2 = %llu\n", (unsigned long long int) N*(N-1)/2);
 
    // Release device memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
 
    // Release host memory
    free(h_a); free(h_b); free(h_c);
 
    return 0;
}
