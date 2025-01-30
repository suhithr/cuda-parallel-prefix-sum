/* Blelloch Scan Prefix Sum Parallel Implementation */
#include "check_cuda.h"
#include <iostream>
#include <bit>
#include <cmath>

/* This version will have many shared memory bank conflicts leading
to inefficiency*/
__device__ int ilog2_cuda(size_t n)
{
    return (n == 0) ? -1 : (8 * sizeof(n) - __clz(n) - 1);
}

__global__ void blellochScanKernel(const uint32_t *const in, uint32_t *out, const size_t len)
{
    uint32_t global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t local_i = threadIdx.x;
    extern __shared__ uint32_t sharedMem[];

    // Copy to shared memory and zero it out if needed
    if (global_tid < int(len))
    {
        sharedMem[local_i] = in[global_tid];
    }
    else
    {
        sharedMem[local_i] = 0.0f;
    }
    __syncthreads();
    
    /* Up Sweep portion
        We build a reduction tree of partial sums
    */
    int stride = 1;
    // int log2_len = ilog2_cuda(len);
    // d actually indicates how many threads run at each stage
    for (int d = len >> 1; d > 0; d>>= 1)
    {
        if (local_i < d)
        {
            /* thread mapping allows us to smartly map a series of consecutive threads to non consecutive
            elements.
            These expressions will return elements as pairs in constructing the tree*/
            int ai = stride * (2 * local_i + 1) - 1;
            int bi = stride * (2 * local_i + 2) - 1;
            sharedMem[bi] = sharedMem[ai] + sharedMem[bi]; 
        }
        stride <<= 1;
        __syncthreads();
    }

    // /* Down Sweep Portion*/
    if (local_i == 0)
    {
        sharedMem[len-1] = 0;
    }
    __syncthreads();
    // stride = len >> 1;
    for (int d = 1; d < len; d <<= 1)
    {
        if (local_i < d)
        {
            int ai = stride * (2 * local_i + 1) - 1;
            int bi = stride * (2 * local_i + 2) - 1;
            // uncomment to visualize the threads getting mapped to indices
            // printf("Threadid %d ai %d :: bi %d\n ", threadIdx.x, ai, bi);
            int temp = sharedMem[bi];
            sharedMem[bi] = sharedMem[ai] + sharedMem[bi];
            sharedMem[ai] = temp;
        }
        stride >>= 1;
        __syncthreads();
    }

    if (global_tid < len)
    {
        out[global_tid] = sharedMem[local_i];
    }
}
void blellochScan(const uint32_t* const in, uint32_t* out, const size_t len)
{
    const uint32_t THREADS_PER_BLOCK = 1024;

    uint32_t *in_d, *out_d; 
    checkCudaErrors(cudaMalloc((void **) &in_d, len * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void **) &out_d, len * sizeof(uint32_t)));
    // checkCudaErrors(cudaMalloc((void **) &block_sum_d, 1 * sizeof(uint32_t)));

    checkCudaErrors(cudaMemcpy(in_d, in, len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    blellochScanKernel<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(uint32_t)>>>(in_d, out_d, len);

    checkCudaErrors(cudaMemcpy(out, out_d, len * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(in_d));
    checkCudaErrors(cudaFree(out_d));
}