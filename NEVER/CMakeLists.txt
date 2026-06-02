cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(NEVER VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89") # Targeting Turing, Ampere, Ada

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(SOURCES
    src/main.cu
)

# Executable
add_executable(never ${SOURCES})

# Target properties
set_target_properties(never PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(never PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xptxas -O3 -Xcompiler -O3>
)
