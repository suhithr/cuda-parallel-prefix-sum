#ifndef BLELLOCH_STONE_SCAN_H__
#define BLELLOCH_STONE_SCAN_H__

#include "cuda_runtime.h"

#include <iostream>

void blellochScan(const uint32_t *const in, uint32_t *out, const size_t len);

#endif