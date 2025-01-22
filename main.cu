#include <iostream>
#include <random>
#include <cassert>
#include <sstream>

#include "knogge_stone_scan.h"
#include "profile_function.h"


void cpu_prefix_scan(const uint32_t* const in, uint32_t* const out, const size_t len)
{
    uint32_t run_sum = 0;
    for (int i = 0; i < len; ++i)
    {
        run_sum  = run_sum + in[i];
        out[i] = run_sum;
    }
}

void assertArrayEqual(const uint32_t* arr1, const uint32_t* arr2, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        assert (arr1[i] == arr2[i]);
        
    }
}

int main()
{
    const size_t INPUT_LENGTH = 10;
    // Generate random input
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100000);

    uint32_t* in = new uint32_t[INPUT_LENGTH];
    for (int i = 0; i < INPUT_LENGTH; ++i)
    {
        in[i] = i; //dist(gen);
    }
    std::cout << "Input : ";
    for (int i = 0; i < INPUT_LENGTH; i++)
    {
        std::cout << " " << in[i];
    }
    std::cout << "\n";

    // Create host memory for output
    uint32_t* out  = new uint32_t[INPUT_LENGTH];

    // Do CPU scan
    std::cout << "Performing CPU Scan" << std::endl;

    cpu_prefix_scan(in, out, INPUT_LENGTH);
    for (int i = 0; i < INPUT_LENGTH; i++)
    {
        std::cout << " " << out[i];
    }
    std::cout << "\n";

    // Create host memory for output
    uint32_t* out_gpu  = new uint32_t[INPUT_LENGTH];
    // Do GPU scan
    std::cout << "Performing Knogge-Stone Scan" << std::endl;
    prefixScan(in, out_gpu, INPUT_LENGTH);

    assertArrayEqual(out, out_gpu, INPUT_LENGTH);
    for (int i = 0; i < INPUT_LENGTH; i++)
    {
        std::cout << " " << out_gpu[i];
    }

    
}