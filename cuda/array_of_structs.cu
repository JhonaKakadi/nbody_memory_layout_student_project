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
	const float inv_dist_cube = 1.0f / sqrt(dist_sixth);		// use cudaSqrt for performance, double is problematic
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
	int id = LINEAR_ID;
	if (id < PROBLEMSIZE) {
		for (int j = 0; j < PROBLEMSIZE; ++j) {
			aos_pp_interaction(&particles[id], &particles[j]);
			//aos_pp_interaction(&particles[j], &particles[id]);
		}
	}
}



// iterates through the particles-array
// and for each (of three) particle calculates the new position.
//
__global__ void aos_move(particle* particles) {
	int id = LINEAR_ID;
	if (id < PROBLEMSIZE) {
		for (int j = 0; j < 3; ++j) {
			particles[id].pos[j] += particles[id].vel[j] * TIMESTEP;
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
	struct particle particles_host[PROBLEMSIZE];
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
	int datasize = sizeof(particles_host);
	
	HANDLE_ERROR( cudaMalloc(&particles_device, datasize) );

	HANDLE_ERROR( cudaMemcpy(particles_device, particles_host, datasize, cudaMemcpyHostToDevice) );
	
	
	
	// loop with STEPS to GPU-compute
	//		-> update
	//		-> run
	// with time measurement (events)
	
	const int num_threads = 1024;
	const int num_SMs = PROBLEMSIZE/num_threads;

    float sum_move = 0, sum_update =0;
    float time_update, time_move;
	for (int s = 0; s < STEPS; ++s) {
		
		cudaEventRecord(start_update, 0);
		aos_update<<< num_SMs, num_threads >>>(particles_device);
		HANDLE_LAST_ERROR;
		cudaEventRecord(stop_update, 0);
		
		cudaEventRecord(start_move, 0);
		aos_move<<< num_SMs, num_threads >>>(particles_device);
		HANDLE_LAST_ERROR;
		cudaEventRecord(stop_move, 0);
		
		cudaEventSynchronize(stop_move);
		cudaEventElapsedTime(&time_update, start_update, stop_update);
		cudaEventElapsedTime(&time_move, start_move, stop_move);
		printf("AoS\t%fms\t%fms\n", time_update, time_move);
        sum_move += time_move;
        sum_update += time_update;
    }
    printf("AVG:\t%3.4fms\t%3.6fms\n\n", sum_update / STEPS, sum_move / STEPS);

	
	
	
	// actually we don't need the results
	// but if we did: copy back
	//HANDLE_ERROR( cudaMemcpy(praticles_host, particles_device, datasize, cudaMemcpyDeviceToHost) );
	
	// compute time for the benchmark
	//float time_update, time_move;
	//cudaEventElapsedTime(&time_update, start_update, stop_update);
	//cudaEventElapsedTime(&time_move, start_move, stop_move);
	
	// print time
	//printf("AoS\t%fms\t%fms\n", time_update, time_move);
	
	
	
	// clean up
	//delete[] particles_host;
	HANDLE_ERROR( cudaFree(particles_device) );	
	HANDLE_ERROR( cudaDeviceReset() );
}