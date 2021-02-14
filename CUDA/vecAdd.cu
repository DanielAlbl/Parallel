/* Introduction code to CUDA
 * Final version, each block 256 threads,
 * each grid having n/256 blocks in it.
 * 
 * Compile: nvcc -g -o vec_add vecAdd2.cu -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
// Kernel function to add the elements of two arrays
__global__ void vecAdd(double *a, double *b, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i += stride)
       b[i] = a[i] + b[i];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 1<<20;

    // Device input vectors
    double *d_a;
    double *d_b;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&d_a, bytes);
    cudaMallocManaged(&d_b, bytes);
 
    // Initialize vectors on host
    for(int i = 0; i < n; i++ ) {
        d_a[i] = sin(i)*sin(i);
        d_b[i] = cos(i)*cos(i);
    }
 
    // Number of threads in each thread block
    int blockSize = 256;
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, n);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();
 
    /* Sum up vector d_b and print result divided by n, 
       this should equal 1 within error */
    double sum = 0.0;
    for(int i=0; i<n; i++)
        sum += d_b[i];
    printf("final result: %f\n", sum/n);
 
    // Release Unified Memory 
    cudaFree(d_a);
    cudaFree(d_b);
 
    return 0;
}
