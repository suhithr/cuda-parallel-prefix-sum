#ifndef KNOGGE_STONE_SCAN_H__
#define KNOGGE_STONE_SCAN_H__

#include "cuda_runtime.h"

#include <iostream>

void prefixScan(const uint32_t *const in,
                uint32_t *out,
                const size_t len);

#endif