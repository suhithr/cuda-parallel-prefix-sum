#ifndef KNOGGE_STONE_SCAN_H__
#define KNOGGE_STONE_SCAN_H__

#include "cuda_runtime.h"

#include <iostream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

void prefixScan(const uint32_t *const in,
                uint32_t *out,
                const size_t len);

#endif