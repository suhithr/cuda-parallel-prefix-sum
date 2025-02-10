cmake_minimum_required(VERSION 3.10)
project(Scan LANGUAGES CXX CUDA)

# Set CUDA and C++ standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Set NVCC options
set(NVCC_OPTS "-arch=sm_60")  # Modify as needed

# Create an optimized build by default
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set up Python for CMake
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Find nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
find_package(nanobind CONFIG REQUIRED)

# Include directories
include_directories(
    ${Python_INCLUDE_DIRS}
    /home/suhith/.local/lib/python3.10/site-packages/nanobind/include
)

# First, create the CUDA executable
set(SOURCES
    main.cu
    knogge_stone_scan.cu
    blelloch_scan.cu
)
add_executable(scan ${SOURCES})
target_compile_options(scan PRIVATE ${NVCC_OPTS})
set_target_properties(scan PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Then, create the Python module
nanobind_add_module(blelloch_scan_module 
    blelloch_scan.cu
)
set_target_properties(blelloch_scan_module PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "blelloch_scan"  # This ensures the output file is named blelloch_scan
)

# Enable CUDA separate compilation for both targets
set_property(TARGET scan PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET blelloch_scan_module PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(scan PRIVATE CUDA::cudart)
target_link_libraries(blelloch_scan_module PRIVATE CUDA::cudart)

# Set the output directory for the Python module
set_target_properties(blelloch_scan_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python
)