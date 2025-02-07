/* Blelloch Scan Prefix Sum Parallel Implementation */
#include "check_cuda.h"
#include <iostream>
#include <bit>
#include <cmath>

/* This version will have many shared memory bank conflicts leading
to inefficiency*/

__global__ void blellochScanKernel(const uint32_t *const in, uint32_t *const block_sums_d, uint32_t *out, const uint32_t len)
{
    uint32_t global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t local_tid = threadIdx.x;
    extern __shared__ uint32_t sharedMem[];
    bool active = global_tid < len;

    // Copy to shared memory and zero it out if needed
    if (active)
    {
        sharedMem[local_tid] = in[global_tid];
    }
    else
    {
        sharedMem[local_tid] = 0;
    }
    __syncthreads();
    
    /* Up Sweep portion
        We build a reduction tree of partial sums
    */
    int stride = 1;
    int max_threads = blockDim.x; 

    // Think of this as just building out a tree. Starting from max_threads/2 pairs all the way up
    for (int d = max_threads >> 1; d > 0; d>>= 1, stride <<= 1)
    {
        // d actually indicates how many threads run at each stage
        if (local_tid < d)
        {
            /* thread mapping allows us to smartly map a series of consecutive threads to non consecutive
            elements.
            These expressions will return elements as pairs in constructing the tree.*/
            int ai = stride * (2 * local_tid + 1) - 1;
            int bi = ai + stride;
            sharedMem[bi] = sharedMem[ai] + sharedMem[bi]; 
        }
        // stride <<= 1;
        __syncthreads();
    }

    // /* Down Sweep Portion*/
    if (local_tid == 0 && max_threads > 0)
    {
        block_sums_d[blockIdx.x] = sharedMem[max_threads - 1];
        sharedMem[max_threads-1] = 0;
    }
    // printf("Here tid %d\n", threadIdx.x);
    __syncthreads();
    stride = max_threads >> 1;
    for (int d = 1; d <= max_threads >> 1; d <<= 1)
    {
        if (local_tid < d)
        {
            int left = stride * (2 * local_tid + 1) - 1;
            int right = left + stride;
            // uncomment to visualize the threads getting mapped to indices
            // printf("Threadid %d ai %d :: bi %d\n ", threadIdx.x, ai, bi);
            // if (right < max_threads || true)
            // {
            int temp = sharedMem[left];
            sharedMem[left] = sharedMem[right];
            sharedMem[right] = temp + sharedMem[right];
            // }
        }
        stride >>= 1;
        __syncthreads();
    }

    if (active)
    {
        out[global_tid] = sharedMem[local_tid];
    }
}
__global__ void distributeBlockSums(const uint32_t *const block_sums_d, uint32_t *out, const uint32_t len)
{
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid < len)
    {
        out[global_tid] += block_sums_d[blockIdx.x];
    }
}

void blellochScan(const uint32_t* const in, uint32_t* out, const size_t len)
{
    const uint32_t THREADS_PER_BLOCK = 1024;

    uint32_t *in_d, *out_d, *block_sum_d; 
    checkCudaErrors(cudaMalloc((void **) &in_d, len * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void **) &out_d, len * sizeof(uint32_t)));

    checkCudaErrors(cudaMemcpy(in_d, in, len * sizeof(uint32_t), cudaMemcpyHostToDevice));

    if (len <= THREADS_PER_BLOCK)
    {
        checkCudaErrors(cudaMalloc((void **) &block_sum_d, 2 * sizeof(uint32_t)));
        checkCudaErrors(cudaMemset(block_sum_d, 0, sizeof(uint32_t)));
        blellochScanKernel<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(uint32_t)>>>(in_d, block_sum_d, out_d, len);
    }
    else
    {
        /* First Scan */
        uint32_t NUM_BLOCKS = (len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        checkCudaErrors(cudaMalloc((void **) &block_sum_d, NUM_BLOCKS * sizeof(uint32_t)));
        checkCudaErrors(cudaMemset(block_sum_d, 0, NUM_BLOCKS * sizeof(uint32_t)));
        std::cout << "Performing first blelloch scan \n";
        blellochScanKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK,  THREADS_PER_BLOCK * sizeof(uint32_t)>>>(in_d, block_sum_d, out_d, len);

        cudaDeviceSynchronize();

        /* Second Scan */


        if (NUM_BLOCKS <= THREADS_PER_BLOCK) {

            // this is just if all the block sums fit in one block
            uint32_t *dummy_block_sums_d;
            checkCudaErrors(cudaMalloc((void **) &dummy_block_sums_d, sizeof(uint32_t)));
            checkCudaErrors(cudaMemset(dummy_block_sums_d, 0, sizeof(uint32_t)));
            std::cout << "Performing second blelloch scan \n";
            blellochScanKernel<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(uint32_t)>>>(block_sum_d, dummy_block_sums_d, block_sum_d, NUM_BLOCKS);

            checkCudaErrors(cudaFree(dummy_block_sums_d));
        }
        else
        {
            blellochScan(block_sum_d, block_sum_d, NUM_BLOCKS);
        }


        distributeBlockSums<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(block_sum_d, out_d, len); 

        uint32_t* block_sum_host = new uint32_t[NUM_BLOCKS];
        checkCudaErrors(cudaMemcpy(block_sum_host, block_sum_d, NUM_BLOCKS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        // for (int i = 0; i < NUM_BLOCKS ; i++)
        // {
        //     std::cout << " " << block_sum_host[i];
        // }
        // std::cout << "\n";
    }
    checkCudaErrors(cudaMemcpy(out, out_d, len * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(in_d));
    checkCudaErrors(cudaFree(out_d));
    checkCudaErrors(cudaFree(block_sum_d));
}