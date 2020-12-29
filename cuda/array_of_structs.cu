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



__global__ void aos_update_t(particle* particles) {
    int id = LINEAR_ID;		// thread id
    // iterate through all chunks
    for (int other = 0; other < PROBLEMSIZE; ++other) {
        // loop through a chunk
        aos_pp_interaction(&particles[id], &particles[other]);
    }
}


// iterates through the particles-matrix
// and lets compute the interaction between two
// particles respectively.
//
__global__ void aos_update_t_shared(particle* particles) {\
	int id = LINEAR_ID;		// thread id
	int idx = threadIdx.x;
	const int chunksize = 1024;

	__shared__ particle particle_chunk[chunksize];

	// iterate through all chunks
	for (int chunk = 0; chunk < (PROBLEMSIZE / chunksize); ++chunk) {
		// update chunk
		particle_chunk[idx] = particles[idx + chunksize * chunk];
		SYNC_THREADS;

		// loop through a chunk
		for (int p = 0; p < chunksize; p++) {
			aos_pp_interaction(&particles[id], &particle_chunk[p]);
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



__global__ void aos_randNormal(struct particle* particle) {
	int tid = LINEAR_ID;
	curandState state;
	curand_init(1337, tid, 0, &state);

	// write random values to particle attributes
	struct particle p = particle[tid];

	for (int i = 0; i < 3; ++i) {
		p.pos[i] = curand_normal(&state);
		p.vel[i] = curand_normal(&state) / 10.0f;
	}
	p.mass = curand_normal(&state) / 100.0f;
}



void aos_run(void) {

	// init arrays
	struct particle* particles_host = (particle*) malloc(PROBLEMSIZE * sizeof(particle));
	struct particle* particles_device;

	// init kernel parameters
	const int num_threads = 1024;
	const int num_blocks = PROBLEMSIZE/num_threads;

	unsigned long long datasize = PROBLEMSIZE*sizeof(particle);
	HANDLE_ERROR( cudaMalloc(&particles_device, datasize) );

	// fill particle_device with random values
	aos_randNormal<<< num_blocks, num_threads >>>(particles_device);
	cudaDeviceSynchronize();
	HANDLE_LAST_ERROR;




	// init event parameters
    float sum_move = 0, sum_update = 0 , sum_update_shared=0;
    float time_update_t, time_update_t_shared, time_move;
	
	// loop with STEPS to GPU-compute
	//		-> update
	//		-> run
	// with time measurement (events)
	
	// init event management
	cudaEvent_t start_update_t, stop_update_t;
	cudaEventCreate(&start_update_t);
	cudaEventCreate(&stop_update_t);
	cudaEvent_t start_update_t_shared, stop_update_t_shared;
	cudaEventCreate(&start_update_t_shared);
	cudaEventCreate(&stop_update_t_shared);

	cudaEvent_t start_move, stop_move;
	cudaEventCreate(&start_move);
	cudaEventCreate(&stop_move);
	
	for (int s = 0; s < STEPS; ++s) {

		cudaEventRecord(start_update_t, 0);
		aos_update_t<<< num_blocks, num_threads >>>(particles_device);
		HANDLE_LAST_ERROR;
		cudaEventRecord(stop_update_t, 0);
		
		cudaEventRecord(start_update_t_shared, 0);
		aos_update_t_shared<<< num_blocks, num_threads >> > (particles_device);
		HANDLE_LAST_ERROR;
		cudaEventRecord(stop_update_t_shared, 0);


		cudaEventRecord(start_move, 0);
		aos_move<<< num_blocks, num_threads >>>(particles_device);
		HANDLE_LAST_ERROR;
		cudaEventRecord(stop_move, 0);
		
		cudaEventSynchronize(stop_move);
		cudaEventElapsedTime(&time_update_t, start_update_t, stop_update_t);
		cudaEventElapsedTime(&time_update_t_shared, start_update_t_shared, stop_update_t_shared);
		cudaEventElapsedTime(&time_move, start_move, stop_move);
		printf("AoS\t%fms\t%fms\t%fms\n", time_update_t, time_update_t_shared, time_move);
        sum_move += time_move;
        sum_update += time_update_t;
        sum_update_shared += time_update_t_shared;
    }
    printf("AVG:\t%3.4fms\t%3.4fms\t%3.6fms\n\n", sum_update / STEPS, sum_update_shared/ STEPS, sum_move / STEPS);

	
	
	// actually we don't need the results
	// but if we did: copy back
	//HANDLE_ERROR( cudaMemcpy(praticles_host, particles_device, datasize, cudaMemcpyDeviceToHost) );
	
	
	
	// clean up
	free(particles_host);
	HANDLE_ERROR( cudaFree(particles_device) );	
	HANDLE_ERROR( cudaDeviceReset() );
}