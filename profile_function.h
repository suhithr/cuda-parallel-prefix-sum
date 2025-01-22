/* Profiling Function */
#ifndef PROFILE_FUNCTION_H
#define PROFILE_FUNCTION_H

#include <iostream>
#include <chrono>
#include <functional>
#include <utility>

template <typename Func, typename... Args>
void profile_function(const std::string& name, Func&& func, Args&&... args)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "Function [" << name << "] took " << duration.count() << " ms.\n";
}

#endif