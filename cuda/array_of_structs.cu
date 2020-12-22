#include <stdio.h>
#include <math.h>
#include "shared_header.h"

struct particle {
	float pos[3];
	float vel[3];
	float mass;
};



__device__ inline void aos_pp_interaction(struct particle* p_i, struct particle* p_j) {
	float distance_sqr_x = (p_i->pos[0] - p_j->pos[0]) * (p_i->pos[0] - p_j->pos[0]);
	float distance_sqr_y = (p_i->pos[1] - p_j->pos[1]) * (p_i->pos[1] - p_j->pos[1]);
	float distance_sqr_z = (p_i->pos[2] - p_j->pos[2]) * (p_i->pos[2] - p_j->pos[2]);
	
	const float dist_sqr = EPS2 + distance_sqr_x + distance_sqr_y + distance_sqr_z;
	const float dist_sixth = dist_sqr * dist_sqr * dist_sqr;
	const float inv_dist_cube = 1.0f / sqrt(double(dist_sixth));		// will there be performance issues with double?
	const float sts = p_j->mass * inv_dist_cube * TIMESTEP;
	
	p_i->vel[0] += distance_sqr_x * sts;
	p_i->vel[1] += distance_sqr_y * sts;
	p_i->vel[2] += distance_sqr_z * sts;
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
	struct particle* particles_device;
	
	// fill array with structs of random values
	for (int i = 0; i < ( sizeof(particles_host) / sizeof(struct particle) ); ++i) {
		for (int j = 0; j < 3; ++j) {
			particles_host[i].pos[j] = (float)rand();
			particles_host[i].vel[j] = (float)rand() / 10.0f;
		}
		particles_host[i].mass = (float)rand() / 100.0f;
		
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
	printf("AoS\t%fms\t%fms\n", time_update, time_move);
	
	
	
	// clean up
	//delete[] particles_host;
	HANDLE_ERROR( cudaFree(particles_device) );	
	HANDLE_ERROR( cudaDeviceReset() );
}