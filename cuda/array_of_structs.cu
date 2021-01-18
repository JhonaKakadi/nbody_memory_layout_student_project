#include <stdio.h>
#include <math.h>
#include "shared_header.h"

struct particle  {
	float pos[3];
	float vel[3];
	float mass;
};


// in this version we are using registers instead of
//__device__ inline void aos_pp_interaction(struct particle* p_i, struct particle* p_j)
//
__device__ inline void aos_pp_interaction(float i_pos_x, float i_pos_y, float i_pos_z, float* i_vel_x, float* i_vel_y, float* i_vel_z,
										  float j_pos_x, float j_pos_y, float j_pos_z, float j_mass) {
	float distance_sqr_x = (i_pos_x - j_pos_x) * (i_pos_x - j_pos_x);
	float distance_sqr_y = (i_pos_y - j_pos_y) * (i_pos_y - j_pos_y);
	float distance_sqr_z = (i_pos_z - j_pos_z) * (i_pos_z - j_pos_z);
	
	const float dist_sqr = EPS2 + distance_sqr_x + distance_sqr_y + distance_sqr_z;
	const float dist_sixth = dist_sqr * dist_sqr * dist_sqr;
	const float inv_dist_cube = 1.0f / sqrt(dist_sixth);		// use cudaSqrt for performance, double is problematic
	const float sts = j_mass * inv_dist_cube * TIMESTEP;
	
	(*i_vel_x) += distance_sqr_x * sts;
	(*i_vel_y) += distance_sqr_y * sts;
	(*i_vel_z) += distance_sqr_z * sts;
}


__global__ void aos_update_t(particle* particles) {
    int id = LINEAR_ID;		// thread id
	
	// get single data from struct particle
	float i_pos_x = particles[id].pos[0];
    float i_pos_y = particles[id].pos[1];
    float i_pos_z = particles[id].pos[2];
	float i_vel_x = particles[id].vel[0];
	float i_vel_y = particles[id].vel[1];
	float i_vel_z = particles[id].vel[2];
	
    // iterate through all chunks
    for (int other = 0; other < PROBLEMSIZE; ++other) {
	
		// get single data from struct particle
		float j_pos_x = particles[other].pos[0];
        float j_pos_y = particles[other].pos[1];
        float j_pos_z = particles[other].pos[2];
		float j_mass = particles[other].mass;
		
		// loop through a chunk
        aos_pp_interaction(i_pos_x, i_pos_y, i_pos_z, &i_vel_x, &i_vel_y, &i_vel_z,
						   j_pos_x, j_pos_y, j_pos_z, j_mass);
						   
		// write back copied values
		particles->vel[0] = i_vel_x;
		particles->vel[1] = i_vel_y;
		particles->vel[2] = i_vel_z;
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
	
	// get single data from struct particle
	float i_pos_x = particles[id].pos[0];
    float i_pos_y = particles[id].pos[1];
    float i_pos_z = particles[id].pos[2];
	float i_vel_x = particles[id].vel[0];
	float i_vel_y = particles[id].vel[1];
	float i_vel_z = particles[id].vel[2];

	// iterate through all chunks
	for (int chunk = 0; chunk < (PROBLEMSIZE / chunksize); ++chunk) {
		// update chunk
		particle_chunk[idx] = particles[idx + chunksize * chunk];
		SYNC_THREADS;

		// loop through a chunk
		for (int p = 0; p < chunksize; p++) {
		
			// get single data from struct particle
			float j_pos_x = particle_chunk[p].pos[0];
			float j_pos_y = particle_chunk[p].pos[1];
			float j_pos_z = particle_chunk[p].pos[2];
			float j_mass = particle_chunk[p].mass;
			
			
			aos_pp_interaction(i_pos_x, i_pos_y, i_pos_z, &i_vel_x, &i_vel_y, &i_vel_z,
						       j_pos_x, j_pos_y, j_pos_z, j_mass);
							   
			// write back copied values
			particles->vel[0] = i_vel_x;
			particles->vel[1] = i_vel_y;
			particles->vel[2] = i_vel_z;
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
	
	particle[tid] = p;
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
	HANDLE_ERROR( cudaDeviceSynchronize() );
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
	HANDLE_ERROR( cudaEventCreate(&start_update_t) );
	HANDLE_ERROR( cudaEventCreate(&stop_update_t) );
	
	cudaEvent_t start_update_t_shared, stop_update_t_shared;
	HANDLE_ERROR( cudaEventCreate(&start_update_t_shared) );
	HANDLE_ERROR( cudaEventCreate(&stop_update_t_shared) );

	cudaEvent_t start_move, stop_move;
	HANDLE_ERROR( cudaEventCreate(&start_move) );
	HANDLE_ERROR( cudaEventCreate(&stop_move) );
	
	printf("Benchmarks: Thread, \tThread_shared, \tmove\n");
	for (int s = 0; s < STEPS; ++s) {

		HANDLE_ERROR( cudaEventRecord(start_update_t, 0) );
		aos_update_t<<< num_blocks, num_threads >>>(particles_device);
		HANDLE_ERROR( cudaEventRecord(stop_update_t, 0) );
		
		HANDLE_ERROR( cudaEventRecord(start_update_t_shared, 0) );
		aos_update_t_shared<<< num_blocks, num_threads >>>(particles_device);
		HANDLE_ERROR( cudaEventRecord(stop_update_t_shared, 0) );

		HANDLE_ERROR( cudaEventRecord(start_move, 0) );
		aos_move<<< num_blocks, num_threads >>>(particles_device);
		HANDLE_ERROR( cudaEventRecord(stop_move, 0) );
		
		HANDLE_ERROR( cudaEventSynchronize(stop_move) );
		HANDLE_ERROR( cudaEventElapsedTime(&time_update_t, start_update_t, stop_update_t) );
		HANDLE_ERROR( cudaEventElapsedTime(&time_update_t_shared, start_update_t_shared, stop_update_t_shared) );
		HANDLE_ERROR( cudaEventElapsedTime(&time_move, start_move, stop_move) );
		
		printf("AoS\t%3.4fms\t%3.4fms\t%3.6fms\n", time_update_t, time_update_t_shared, time_move);
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