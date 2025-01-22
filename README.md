## Parallel Prefix Sum on the GPU

This repository contains implementations of various parallel prefix sum algorithms on the GPU.

#### Knogge-Stone Scan Algorithm
Implements a parallel inclusive scan of an input array of `uint32_t` elements.

The kernel is launched on a single block, of 1024 threads. Thus it only works on arrays up to 1024 elements.