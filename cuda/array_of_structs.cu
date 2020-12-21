#include <stdio.h>
#include "shared_header.h"

struct particle {
	float pos[3];
	float vel[3];
	float mass;
};

inline void aos_pp_interaction(struct particle* p_i, struct particle* p_j) {
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
			particles[i].pos[j] += particles[i].vel[j] * kTimestep;
		}
	}
}
	

int aos_run(void) {

	// init event management
	cudaEvent_t start_update, stop_update;
	cudaEventCreate(&start_update);
	cudaEventCreate(&stop_update);
	
	cudaEvent_t start_move, stop_move;
	cudaEventCreate(&start_move);
	cudaEventCreate(&stop_move);


	// init array
	struct particle particles[kProblemSize];
	
	// fill array with structs of random values
	//for (int i = 0; i < ( sizeof(particles) / sizeof(struct particle) ); ++i) {
		for (int j = 0; j < 3; ++j) {
			particles[i].pos[j] = 0;
		}
		particles[i].mass = 0;
		
		// temporarily it's still filled with zero
		// todo: find rng with normaldistribution
		// or just use random?
	}
	
	// loop with kSteps to GPU-compute
	//		-> update
	//		-> run
	// with time measurement (events)
	//
	for (int s = 0; s < kSteps; ++s) {
		
		cudaEventRecord(start_update);
		aos_update<<<1,1>>>(particles);
		HANDLE_ERROR(cudaGetLastError());
		cudaEventRecord(stop_update);
		
		cudaEventRecord(start_move);
		aos_move<<<1,1>>>(particles);
		HANDLE_ERROR(cudaGetLastError());
		cudaEventRecord(stop_move);
	}
	cudaEventSynchronize(stop_update);
	cudaEventSynchronize(start_move);
	
	float time_update, time_move;
	cudaEventElapsedTime(&time_update, start_update, stop_update);
	cudaEventElapsedTime(&time_move, start_move, stop_move);
	
	// print time
	printf("AoS\t%f\t%f\n", time_update, time_move);
}