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

void assert_array_equal(const uint32_t* arr1, const uint32_t* arr2, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        assert (arr1[i] == arr2[i]);
        
    }
}

int main()
{
    const size_t INPUT_LENGTH = 1000;
    // Generate random input
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 5000);

    std::cout << "Generating input of length " << INPUT_LENGTH << "\n";
    uint32_t* in = new uint32_t[INPUT_LENGTH];
    for (int i = 0; i < INPUT_LENGTH; ++i)
    {
        in[i] = dist(gen);
    }

    // Create host memory for output
    uint32_t* out  = new uint32_t[INPUT_LENGTH];

    // Do CPU scan
    std::cout << "Performing CPU Scan" << std::endl;

    profile_function("cpu_prefix_scan", cpu_prefix_scan, in, out, INPUT_LENGTH);
    // for (int i = 0; i < INPUT_LENGTH; i++)
    // {
    //     std::cout << " " << out[i];
    // }
    std::cout << "\n";

    // Create host memory for output
    uint32_t* out_gpu  = new uint32_t[INPUT_LENGTH];
    // Do GPU scan
    std::cout << "Performing Knogge-Stone Scan" << std::endl;
    profile_function("prefixScan", prefixScan, in, out_gpu, INPUT_LENGTH);

    assert_array_equal(out, out_gpu, INPUT_LENGTH);
    std::cout << "Verified arrays are equal \n";
    // for (int i = 0; i < INPUT_LENGTH; i++)
    // {
    //     std::cout << " " << out_gpu[i];
    // }

    
}