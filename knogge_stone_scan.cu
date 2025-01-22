/* Knogge-Stone Prefix Sum Parallel Scan Implementation */
#include "knogge_stone_scan.h"
#include <iostream>
#include <math.h>


/* Scan Kernel
Inclusive scan, limited to scan size < threads per block
*/
__global__ void prefixScanKernel(const uint32_t *const in, uint32_t *const out, const size_t len)
{

    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t local_i = threadIdx.x;

    extern __shared__ uint32_t sharedMem[];
    if (i < int(len))
    {
        sharedMem[local_i] = in[i];
    }
    else
    {
        sharedMem[local_i] = 0.0f;
    }
    __syncthreads();
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2)
    {
        float temp;
        __syncthreads();
        if (i >= stride)
        {
            temp = sharedMem[local_i] + sharedMem[local_i - stride];
        }
        __syncthreads();
        if (i >= stride)
        {
            sharedMem[local_i] = temp;
        }
    }
    if (i < len)
    {
        out[i] = sharedMem[local_i];
    }
}

void prefixScan(const uint32_t* const in, uint32_t* out, const size_t len)
{
    const uint32_t THREADS_PER_BLOCK = 1024;

    uint32_t *in_d, *out_d;
    checkCudaErrors(cudaMalloc((void **) &in_d, len * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void **) &out_d, len * sizeof(uint32_t)));

    checkCudaErrors(cudaMemcpy(in_d, in, len * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // launch parameters
    // grid dim, block dim, shared memory space
    prefixScanKernel<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(uint32_t)>>>(in_d, out_d, len);

    checkCudaErrors(cudaMemcpy(out, out_d, len * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(in_d));
    checkCudaErrors(cudaFree(out_d));
}