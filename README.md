# n-body memory layout exploration on GPUs using CUDA

This project continues on the n-body memory exploration on CPUs using C++ by Bernhard M. Gruber, see
https://github.com/bernhardmgruber/nbody_memory_layout_student_project
which is also part of this project (in directory `cpp`).
Our implementation is located in `cuda`.

## Building and running

As in the C++ project the following will work
```bash
cd cuda
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./nbody
```

It is also possible to compile without a makefile using
```bash
cd cuda
nvcc main.cu array_of_structs.cu struct_of_arrays.cu array_of_struct_of_arrays.cu
./nbody
```
