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

    extern __shared__ float sharedMem[];
    if (i < int(len))
    {
        sharedMem[local_i] = in[i];
    }
    else
    {
        sharedMem[local_i] = 0.0f;
    }
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
    cudaMalloc((void **) &in_d, len);
    cudaMalloc((void **) &out_d, len);

    cudaMemcpy(in_d, in, len, cudaMemcpyHostToDevice);
    prefixScanKernel<<<THREADS_PER_BLOCK, 1>>>(in_d, out_d, len);
    cudaMemcpy(out, out_d, len, cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);
}