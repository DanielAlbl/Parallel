/* Introduction code to CUDA
 * Second attempt to push this onto the device
 * 
 * Compile: nvcc -g -o vec_add vecAdd2.cu -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int index = threadIdx.x;
    int stride = blockDim.x;
 
    // Make sure we do not go out of bounds
    for (int i = index; i<n; i += stride)
        c[i] = a[i] + b[i];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 1<<20;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
 
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    // Number of threads in each thread block
    int blockSize = 256;
 
    // Execute the kernel
    vecAdd<<<1, blockSize>>>(d_a, d_b, d_c, n);
 
   // Wait for the GPU to finish
   cudaDeviceSynchronize(); 

   // Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/n);
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
