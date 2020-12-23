# This file handles building each task as sub-project.
#
# Each task becomes an independent project in which you can experiment.
#
# Tasks are added as separate sub-projects so as compilation errors in
# one task don't affect another task.
cmake_minimum_required(VERSION 3.10)
project(nbody CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(nbody main.cu)
target_compile_options(${PROJECT_NAME} PRIVATE
        # -fno-math-errno # sqrt prevents vectorization otherwise
        # -mavx512f # if your system does not support his, use -march=native
        # -O3 #optimize O3 default
        # -m64 # flag for 64 bit architecture
        # --gpu-architecture=sm_50 #set specific architecture 50 is maxwell 60/1/2 is pascal
        # 70/2 is volta 75 is turing 80 is ampere
        # --gpu-architecture=compute_52 --gpu-code=sm_62,sm_70 #better for run on different devices
        # arch sholud be low  code should be high
        # --use_fast_math # implies --ftz=true --prec-div=false --prec-sqrt=false --fmad=true
        # -extra-device-vectorization

        )

set_target_properties(
        nbody
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)