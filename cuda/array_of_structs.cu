#include <stdio.h>
#include <math.h>
#include "shared_header.h"

struct particle  {
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
__global__ void aos_update(particle* particles) {\
	int id = LINEAR_ID;		// thread id
	int idx = threadIdx.x();
	int chunksize = 1024;
	
	__shared__ float particle_chunk[chunksize];
	
	// iterate through all chunks
	for (int chunk = 0; chunk < (PROBLEMSIZE / chunksize); ++chunk) {
		// update chunk
		particle_chunk[idx] = particles[idx + chunksize * chunk]
		SYNC_THREADS;
		
		// loop through a chunk
		for (int p = 0; p < chunksize; p++) {
			aos_pp_interaction(&particle[id], &particles[p]);
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

	// init array
	struct particle* particles_host = (particle*) malloc(PROBLEMSIZE * sizeof(particle));
	struct particle* particles_device;

	// fill array with structs of random values
	for (long i = 0; i < PROBLEMSIZE; ++i) {
		for (int j = 0; j < 3; ++j) {
			particles_host[i].pos[j] = (float)rand();
			particles_host[i].vel[j] = (float)rand() / 10.0f;
		}
		particles_host[i].mass = (float)rand() / 100.0f;
		
		// todo: find rng with normaldistribution
		// or just use random?
	}
	
	
	
	// copy data to device
	unsigned long long datasize = PROBLEMSIZE*sizeof(particle);
	
	HANDLE_ERROR( cudaMalloc(&particles_device, datasize) );

	HANDLE_ERROR( cudaMemcpy(particles_device, particles_host, datasize, cudaMemcpyHostToDevice) );
	
	
	
	// init kernel parameters
	const int num_threads = 1024;
	const int num_blocks = PROBLEMSIZE/num_threads;

	// init event parameters
    float sum_move = 0, sum_update =0;
    float time_update, time_move;
	
	// loop with STEPS to GPU-compute
	//		-> update
	//		-> run
	// with time measurement (events)
	
	// init event management
	cudaEvent_t start_update, stop_update;
	cudaEventCreate(&start_update);
	cudaEventCreate(&stop_update);

	cudaEvent_t start_move, stop_move;
	cudaEventCreate(&start_move);
	cudaEventCreate(&stop_move);
	
	for (int s = 0; s < STEPS; ++s) {
		
		cudaEventRecord(start_update, 0);
		aos_update<<< num_blocks, num_threads >>>(particles_device);
		HANDLE_LAST_ERROR;
		cudaEventRecord(stop_update, 0);
		
		cudaEventRecord(start_move, 0);
		aos_move<<< num_blocks, num_threads >>>(particles_device);
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
	
	
	
	// clean up
	free(particles_host);
	HANDLE_ERROR( cudaFree(particles_device) );	
	HANDLE_ERROR( cudaDeviceReset() );
}

// TODO
// use:
//__device__ float
//curand_normal (curandState_t *state)
// as the following:
/*
curandState *state = 
curand_init(9384, tid, 0, state);
float rand = curand_normal(state);
*/
// taken from
// https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/examples/chapter08/rand-kernel.cu