#include <stdio.h>
#include "shared_header.h"

struct particle {
	float pos[3];
	float vel[3];
	float mass;
};

inline void aos_pp_interaction(struct particle* p_i, struct particle* p_j) {

}


// iterates through half of the particles-matrix
// and lets compute the interaction between two 
// particles respectively.
// both particles will be updated.
//
__global__ void aos_update(particle* particles) {
	for (int i = 0; i < ( sizeof(particles) / sizeof(struct particle) ); ++i) {
		for (int j = i+1; j < ( sizeof(particles) / sizeof(struct particle) ); ++j) {
			aos_pp_interaction(particles[i], particles[j]);
			aos_pp_interaction(particles[j], particles[i]);
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

	// init array
	struct particle particles[kProblemSize];
	
	// fill array with structs of random values
	//for (int i = 0; i < 
	
	// start event 
	
	// loop with kSteps
	//		-> update
	//		-> run
	for (int s = 0; s < kSteps; ++s) {
	
		aos_update<<<1,1>>>(particles);
		HANDLE_ERROR(cudaGetLastError());
		
		aos_move<<<1,1>>>(particles);
		HANDLE_ERROR(cudaGetLastError());
	}
	
	// stop event
	
	// print time
	//printf("AoS\t%f\t%f\n", ..., ...);
}