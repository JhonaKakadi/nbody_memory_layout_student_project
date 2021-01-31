#include "shared_header.h"

#define PBLOCKS_PER_BLOCK 4

struct particle_block        // deleted alignas(64)
{
    struct {
        float x[LANES];
        float y[LANES];
        float z[LANES];
    } pos;
    struct {
        float x[LANES];
        float y[LANES];
        float z[LANES];
    } vel;
    float mass[LANES];
};

__device__ inline void aosoa_pp_interaction(
        float piposx,
        float piposy,
        float piposz,
        float *pivelx,
        float *pively,
        float *pivelz,
        float pjposx,
        float pjposy,
        float pjposz,
        float pjmass) {
    float xdistance = piposx - pjposx;
    float ydistance = piposy - pjposy;
    float zdistance = piposz - pjposz;
    xdistance *= xdistance;
    ydistance *= ydistance;
    zdistance *= zdistance;
    const float distSqr = EPS2 + xdistance + ydistance + zdistance;
    const float distSixth = distSqr * distSqr * distSqr;
    const float invDistCube = 1.0f / sqrt(distSixth);
    const float sts = pjmass * invDistCube * TIMESTEP;
    *pivelx += xdistance * sts;
    *pively += ydistance * sts;
    *pivelz += zdistance * sts;
}

__global__ void aosoa_update_t_shared(particle_block *particles) {
    // TODO check for not multiple of 32
    particle_block* mainBlock;
    // __shared__ mainblocks[4]; // could be useful; test it 
    __shared__ particle_block otherBlock;
    const int mainLane = threadIdx.x % LANES;
    const int offset = threadIdx.x / LANES;
    float x_run = 0.0;
    float y_run = 0.0;
    float z_run = 0.0;
    mainBlock = &(particles[PBLOCKS_PER_BLOCK * blockIdx.x + offset]);
  
    for (int otherBlockIndex = 0; otherBlockIndex < BLOCKS; ++otherBlockIndex) {
        if (threadIdx.x == 0) {
            otherBlock = particles[otherBlockIndex];
        }
        SYNC_THREADS;
        for (int otherLane = 0; otherLane < LANES; ++otherLane) {
            aosoa_pp_interaction(mainBlock->pos.x[mainLane],
                          mainBlock->pos.y[mainLane],
                          mainBlock->pos.z[mainLane],
                          &x_run, &y_run, &z_run,
                          otherBlock.pos.x[otherLane],
                          otherBlock.pos.y[otherLane],
                          otherBlock.pos.z[otherLane],
                          otherBlock.mass[otherLane]);
        }
    }
    mainBlock->vel.x[mainLane] += x_run;
    mainBlock->vel.y[mainLane] += y_run;
    mainBlock->vel.z[mainLane] += z_run;
}

__global__ void aosoa_update_t(particle_block *particles) {
    // TODO check for not multiple of 32
    const int offset = threadIdx.x / LANES;
    const int mainLane = threadIdx.x % LANES;
    particle_block mainBlock = particles[PBLOCKS_PER_BLOCK * blockIdx.x + offset];
    particle_block otherBlock;
    float x_run = 0.0;
    float y_run = 0.0;
    float z_run = 0.0;
    for (int otherBlockIndex = 0; otherBlockIndex < BLOCKS; ++otherBlockIndex) {
        otherBlock = particles[otherBlockIndex];
        for (int otherLane = 0; otherLane < LANES; ++otherLane) {
            aosoa_pp_interaction(mainBlock.pos.x[mainLane],
                          mainBlock.pos.y[mainLane],
                          mainBlock.pos.z[mainLane],
                          &x_run, &y_run, &z_run,
                          otherBlock.pos.x[otherLane],
                          otherBlock.pos.y[otherLane],
                          otherBlock.pos.z[otherLane],
                          otherBlock.mass[otherLane]);
        }
    }
    mainBlock.vel.x[mainLane] += x_run;
    mainBlock.vel.y[mainLane] += y_run;
    mainBlock.vel.z[mainLane] += z_run;
}

// Todo find a way to use more threads per block and less blocks
__global__ void aosoa_move(struct particle_block *particle_block) {
    int block_index = blockIdx.x;
    int i = threadIdx.x;

    struct particle_block block = particle_block[block_index];
    block.pos.x[i] += block.vel.x[i] * TIMESTEP;
    block.pos.y[i] += block.vel.y[i] * TIMESTEP;
    block.pos.z[i] += block.vel.z[i] * TIMESTEP;
}

__global__ void aosoa_randNormal(particle_block* particle_blocks) {
    int block = blockIdx.x;
    int lane = threadIdx.x;
    curandState state;
    curand_init(1337, lane, 0, &state);
  
    // iterate over pos-array inside the struct, then over the vel-arrays, then mass-array
    particle_blocks[block].pos.x[lane] = curand_normal(&state);
    particle_blocks[block].pos.y[lane] = curand_normal(&state);
    particle_blocks[block].pos.z[lane] = curand_normal(&state);
    particle_blocks[block].vel.x[lane] = curand_normal(&state) / 10.0f;
    particle_blocks[block].vel.y[lane] = curand_normal(&state) / 10.0f;
    particle_blocks[block].vel.z[lane] = curand_normal(&state) / 10.0f;

    particle_blocks[block].mass[lane] = curand_normal(&state) / 100.0f;
}


void aosoa_run() {

    // "allocate" mem
    struct particle_block* particle_block_host = (particle_block*) malloc(BLOCKS* sizeof(particle_block));
    struct particle_block *particle_block_device;

    // TODO Corrrect omitted particels if PROBLEMSIZE not multiple of LANES
    // fill with random values
    // iterate over the structs 'stru' in the array
    /*
    for (int stru = 0; stru < BLOCKS; ++stru) {
        // iterate over pos-array inside the struct, then over the vel-arrays, then mass-array
        for (int l = 0; l < LANES; ++l) {
            particle_block_host[stru].pos.x[l] = (float) rand();
            particle_block_host[stru].pos.y[l] = (float) rand();
            particle_block_host[stru].pos.z[l] = (float) rand();

            particle_block_host[stru].vel.x[l] = (float) rand() / 10.0f;
            particle_block_host[stru].vel.y[l] = (float) rand() / 10.0f;
            particle_block_host[stru].vel.z[l] = (float) rand() / 10.0f;

            particle_block_host[stru].mass[l] = (float) rand() / 100.0f;
        }
    }*/

   

    int datasize = BLOCKS*sizeof(particle_block);
    HANDLE_ERROR(cudaMalloc(&particle_block_device, datasize));
    // HANDLE_ERROR(cudaMemcpy(particle_block_device, particle_block_host, datasize, cudaMemcpyHostToDevice));
    aosoa_randNormal <<< BLOCKS, LANES >>> (particle_block_device);
    // init event management
    cudaEvent_t start_update, stop_update;
    HANDLE_ERROR(cudaEventCreate(&start_update));
    HANDLE_ERROR(cudaEventCreate(&stop_update));

    cudaEvent_t start_update_shared, stop_update_shared;
    HANDLE_ERROR(cudaEventCreate(&start_update_shared));
    HANDLE_ERROR(cudaEventCreate(&stop_update_shared));
    
    cudaEvent_t start_move, stop_move;
    HANDLE_ERROR(cudaEventCreate(&start_move));
    HANDLE_ERROR(cudaEventCreate(&stop_move));
    float sum_move = 0, sum_update = 0, sum_update_shared = 0;
    float time_update, time_update_shared, time_move;
	
	printf("Benchmarks: Thread, \tThread_shared, \tmove\n");
    for (int i = 0; i < STEPS; ++i) {
        // call update
        HANDLE_ERROR(cudaEventRecord(start_update, 0));
        aosoa_update_t<<<(PROBLEMSIZE + (PBLOCKS_PER_BLOCK * LANES) - 1) / (PBLOCKS_PER_BLOCK * LANES), PBLOCKS_PER_BLOCK *LANES>>>(particle_block_device);
        HANDLE_ERROR(cudaEventRecord(stop_update, 0));
        
        HANDLE_ERROR(cudaEventRecord(start_update_shared, 0));
        aosoa_update_t_shared<<<(PROBLEMSIZE + (PBLOCKS_PER_BLOCK *LANES) - 1) / (PBLOCKS_PER_BLOCK *LANES), PBLOCKS_PER_BLOCK*LANES>>>(particle_block_device);
        HANDLE_ERROR(cudaEventRecord(stop_update_shared, 0));

        // call move
        HANDLE_ERROR(cudaEventRecord(start_move, 0));
        aosoa_move << <(PROBLEMSIZE + LANES - 1) / LANES, LANES >>> (particle_block_device);
        HANDLE_ERROR(cudaEventRecord(stop_move));

        HANDLE_ERROR(cudaEventSynchronize(stop_move));
        HANDLE_ERROR(cudaEventElapsedTime(&time_update, start_update, stop_update));
        HANDLE_ERROR(cudaEventElapsedTime(&time_update_shared, start_update_shared, stop_update_shared));
        HANDLE_ERROR(cudaEventElapsedTime(&time_move, start_move, stop_move));
        printf("AoSoA\t%3.4fms\t%3.4fms\t%3.6fms\n", time_update,time_update_shared, time_move);
        sum_move += time_move;
        sum_update += time_update;
        sum_update_shared += time_update_shared;
    }
    printf("AVG:\t%3.4fms\t%3.4fms\t%3.6fms\n\n", sum_update / STEPS, sum_update_shared / STEPS,  sum_move / STEPS);

    // maybe write back

    // free mem
    free(particle_block_host);
    HANDLE_ERROR(cudaFree(particle_block_device));
    HANDLE_ERROR(cudaDeviceReset());
}
