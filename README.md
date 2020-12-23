# n-body memory layout exploration on GPUs using CUDA

This project continues on the n-body memory exploration on CPUs using C++ by Bernhard M. Gruber, see
https://github.com/bernhardmgruber/nbody_memory_layout_student_project

## Building and running

As in the C++ project the following will work
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./nbody
```

It is also possible to compile without a makefile using
```bash
nvcc main.cu array_of_structs.cu struct_of_arrays.cu array_of_struct_of_arrays.cu
./nbody
```
