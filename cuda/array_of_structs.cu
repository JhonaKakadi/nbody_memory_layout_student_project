#include <stdio.h>
#include "shared_header.h"

struct particle {
	float pos[3];
	float vel[3];
	float mass;
};



__device__ inline void aos_pp_interaction(struct particle* p_i, struct particle* p_j) {
	// still not sure if i want to stick with the data structure
	// so i didn't implement it yet
}



// iterates through half of the particles-matrix
// and lets compute the interaction between two 
// particles respectively.
// both particles will be updated.
//
__global__ void aos_update(particle* particles) {
	for (int i = 0; i < ( sizeof(particles) / sizeof(struct particle) ); ++i) {
		for (int j = i+1; j < ( sizeof(particles) / sizeof(struct particle) ); ++j) {
			aos_pp_interaction(&particles[i], &particles[j]);
			aos_pp_interaction(&particles[j], &particles[i]);
		}
	}
}



// iterates through the particles-array
// and for each (of three) particle calculates the new position.
//
__global__ void aos_move(particle* particles) {
	for (int i = 0; i < ( sizeof(particles) / sizeof(struct particle) ); ++i) {
		for (int j = 0; j < 3; ++j) {
			particles[i].pos[j] += particles[i].vel[j] * TIMESTEP;
		}
	}
}
	


void aos_run(void) {

	// init event management
	cudaEvent_t start_update, stop_update;
	cudaEventCreate(&start_update);
	cudaEventCreate(&stop_update);
	
	cudaEvent_t start_move, stop_move;
	cudaEventCreate(&start_move);
	cudaEventCreate(&stop_move);



	// init array
	struct particle particles_host[kProblemSize];
	struct particle* particles_device[kProblemSize];
	
	// fill array with structs of random values
	for (int i = 0; i < ( sizeof(particles_host) / sizeof(struct particle) ); ++i) {
		for (int j = 0; j < 3; ++j) {
			particles_host[i].pos[j] = 0;
		}
		particles_host[i].mass = 0;
		
		// temporarily it's still filled with zero
		// todo: find rng with normaldistribution
		// or just use random?
	}
	
	
	
	// copy data to device
	int datasize = kProblemSize * sizeof(struct particle);
	
	HANDLE_ERROR( cudaMalloc(&particles_device, datasize) );

	HANDLE_ERROR( cudaMemcpy(particles_device, particles_host, datasize, cudaMemcpyHostToDevice) );
	
	
	
	// loop with kSteps to GPU-compute
	//		-> update
	//		-> run
	// with time measurement (events)
	
	int numSMs;
	HANDLE_ERROR( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0) );
	
	for (int s = 0; s < kSteps; ++s) {
		
		cudaEventRecord(start_update);
		aos_update<<< numSMs, 256 >>>(particles_device);
		HANDLE_ERROR(cudaGetLastError());
		cudaEventRecord(stop_update);
		
		cudaEventRecord(start_move);
		aos_move<<< numSMs, 256 >>>(particles_device);
		HANDLE_ERROR(cudaGetLastError());
		cudaEventRecord(stop_move);
	}
	cudaEventSynchronize(stop_update);
	cudaEventSynchronize(start_move);
	
	
	
	// actually we don't need the results
	// but if we did: copy back
	//HANDLE_ERROR( cudaMemcpy(praticles_host, particles_device, datasize, cudaMemcpyDeviceToHost) );
	
	// compute time for the benchmark
	float time_update, time_move;
	cudaEventElapsedTime(&time_update, start_update, stop_update);
	cudaEventElapsedTime(&time_move, start_move, stop_move);
	
	// print time
	printf("AoS\t%f\t%f\n", time_update, time_move);
	
	
	
	// clean up
	//delete[] particles_host;
	HANDLE_ERROR( cudaFree(particles_device) );	
	HANDLE_ERROR( cudaDeviceReset() );
}