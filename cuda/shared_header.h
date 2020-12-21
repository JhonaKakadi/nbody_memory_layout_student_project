// contains global constants
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


const int kProblemSize = 16 * 1024;
const int kSteps = 5;
const float kTimestep = 0.0001f;
const float kEPS2 = 0.01f;

#define PROBLEMSIZE 16 * 1024
#define STEPS  5
#define TIMESTEP 0.0001f 
#define EPS2 0.01f 
// contains definition of error handler

#define HANDLE_LAST_ERROR handleCudaError(cudaGetLastError(),__FILE__, __LINE__);
#define NUM_OF_THREADS_PER_BLOCK blockDim.x * blockDim.y * blockDim.z
#define TOTAL_NUM_OF_THREADS NUM_OF_THREADS_PER_BLOCK * gridDim.x * gridDim.y * gridDim.z

// Only for 3D-Blocks and 2D-Grids
#define LINEAR_ID threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * threadIdx.z +  NUM_OF_THREADS_PER_BLOCK* blockIdx.x +  NUM_OF_THREADS_PER_BLOCK * gridDim.x *blockIdx.y

#define SYNC_THREADS __syncthreads()

#define HANDLE_ERROR(err)\
	(handleCudaError(err, __FILE__, __LINE__))

static void handleCudaError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("[%s] in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}