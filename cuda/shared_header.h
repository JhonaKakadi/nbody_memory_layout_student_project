#ifndef SHARED_HEADER_H
#define SHARED_HEADER_H

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


// contains global constants

const int kProblemSize = 16 * 1024;
const int kSteps = 5;
const float kTimestep = 0.0001f;
const float kEPS2 = 0.01f;

#define PROBLEMSIZE 16 * 1024
#define STEPS  5
#define TIMESTEP 0.0001f 
#define EPS2 0.01f


// contains definition of error handler

#define HANDLE_LAST_ERROR handleCudaError(cudaGetLastError(),__FILE__, __LINE__)
#define NUM_OF_THREADS_PER_BLOCK blockDim.x * blockDim.y * blockDim.z
#define TOTAL_NUM_OF_THREADS NUM_OF_THREADS_PER_BLOCK * gridDim.x * gridDim.y * gridDim.z

// Only for 3D-Blocks and 2D-Grids
#define LINEAR_ID threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z +  NUM_OF_THREADS_PER_BLOCK* blockIdx.x +  NUM_OF_THREADS_PER_BLOCK * gridDim.x *blockIdx.y

#define SYNC_THREADS __syncthreads()

#define HANDLE_ERROR(err)\
    (handleCudaError(err, __FILE__, __LINE__))

static void handleCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("[%s] in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

void print_cuda_infos(int device, bool mem_props) {
    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, device));
    // general Infos
    int version[2];
    version[0] = props.major;
    version[1] = props.minor;
    int sms = props.multiProcessorCount;

    printf("Name of Device: %s\n", props.name);
    printf("CC-Version is: %d.%d\n", version[0], version[1]);
    printf("SM-Count: %d\n", sms);

    // Block and Thread Infos-Limits
    printf("Warp-size: %d\n", props.warpSize);
    printf("One Block have %d Registers and one SM can handle %d Threads\n", props.regsPerBlock,
           props.maxThreadsPerMultiProcessor);
    printf("One Block can handle: %d Threads\n", props.maxThreadsPerBlock);
    printf("While one Block can have the dimensions: X = %d Y = %d Z = %d\n", props.maxThreadsDim[0],
           props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("And the grid can have the form of: X = %d Y = %d Z = %d\n", props.maxGridSize[0], props.maxGridSize[1],
           props.maxGridSize[2]);
    int *blockDims;
    //blockDims = &props.maxThreadsDim;
    int gridDims[3];
    //gridDims = props.maxGridSize;
    if (mem_props) {
        //print_mem_props(&props);
    }

    //Stream
    // props.deviceoverlap

}


void print_cuda_infos_at_start() {
    int devicecount;
    HANDLE_ERROR(cudaGetDeviceCount(&devicecount));
    if (devicecount == 0) {
        printf("Error there is no cuda device!\n");
        return;
    }
    for (int i = 0; i < devicecount; i++) {
        print_cuda_infos(i, false);
    }
    cudaSetDevice(0);
}

#endif