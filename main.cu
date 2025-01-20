#include <iostream>


void cpu_prefix_scan(const uint32_t* const in, uint32_t* const out, const size_t len)
{
    uint32_t run_sum = 0;
    for (int i = 0; i < len; ++i)
    {
        out[i] = run_sum;
        run_sum  = run_sum + in[i];
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
    cpu_prefix_scan(in, out, INPUT_LENGTH);
    for (int i = 0; i < INPUT_LENGTH; i++)
    {
        std::cout << " " << in[i];
    }
    
}