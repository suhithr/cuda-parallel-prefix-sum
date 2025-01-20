#include <iostream>

#include "knogge_stone_scan.h"


void cpu_prefix_scan(const uint32_t* const in, uint32_t* const out, const size_t len)
{
    uint32_t run_sum = 0;
    for (int i = 0; i < len; ++i)
    {
        run_sum  = run_sum + in[i];
        out[i] = run_sum;
    }
}

int main()
{
    const size_t INPUT_LENGTH = 10;
    // Generate input
    uint32_t* in = new uint32_t[INPUT_LENGTH];
    for (int i = 0; i < INPUT_LENGTH; ++i)
    {
        in[i] =i;
    }

    // Create host memory for output
    uint32_t* out  = new uint32_t[INPUT_LENGTH];

    // Do CPU scan
    std::cout << "Performing CPU Scan" << std::endl;
    cpu_prefix_scan(in, out, INPUT_LENGTH);
    for (int i = 0; i < INPUT_LENGTH; i++)
    {
        std::cout << " " << out[i];
    }

    // Create host memory for output
    uint32_t* out_gpu  = new uint32_t[INPUT_LENGTH];
    // Do GPU scan
    std::cout << "Performing Knogge-Stone Scan" << std::endl;
    prefixScan(in, out_gpu, INPUT_LENGTH);
    for (int i = 0; i < INPUT_LENGTH; i++)
    {
        std::cout << " " << out[i];
    }

    
}