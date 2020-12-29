#include "shared_header.h"

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

__device__ inline void pPInteraction(
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

__global__ void aosoa_update_with_shared(particle_block *particles) {
    // TODO check for not multiple of 32
    __shared__ particle_block mainBlock;
    __shared__ particle_block otherBlock;
    const int mainLane = threadIdx.x;
    if (threadIdx.x == 0) {
        mainBlock = particles[blockIdx.x];
    }
    for (int otherBlockIndex = 0; otherBlockIndex < BLOCKS; ++otherBlockIndex) {
        SYNC_THREADS;
        otherBlock = particles[otherBlockIndex];
        for (int otherLane = 0; otherLane < LANES; ++otherLane) {
            pPInteraction(mainBlock.pos.x[mainLane],
                          mainBlock.pos.y[mainLane],
                          mainBlock.pos.z[mainLane],
                          &mainBlock.vel.x[mainLane],
                          &mainBlock.vel.y[mainLane],
                          &mainBlock.vel.z[mainLane],
                          otherBlock.pos.x[otherLane],
                          otherBlock.pos.y[otherLane],
                          otherBlock.pos.z[otherLane],
                          otherBlock.mass[otherLane]);
        }
    }
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


void aosoa_run() {

    // "allocate" mem
    struct particle_block* particle_block_host = (particle_block*) malloc(BLOCKS* sizeof(particle_block));
    struct particle_block *particle_block_device;

    // TODO Corrrect omitted particels if PROBLEMSIZE not multiple of LANES
    // fill with random values
    // iterate over the structs 'stru' in the array
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
    }

    int datasize = BLOCKS*sizeof(particle_block);
    HANDLE_ERROR(cudaMalloc(&particle_block_device, datasize));
    HANDLE_ERROR(cudaMemcpy(particle_block_device, particle_block_host, datasize, cudaMemcpyHostToDevice));

    // init event management
    cudaEvent_t start_update, stop_update;
    HANDLE_ERROR(cudaEventCreate(&start_update));
    HANDLE_ERROR(cudaEventCreate(&stop_update));

    cudaEvent_t start_move, stop_move;
    HANDLE_ERROR(cudaEventCreate(&start_move));
    HANDLE_ERROR(cudaEventCreate(&stop_move));
    float sum_move = 0, sum_update = 0;
    float time_update, time_move;
    for (int i = 0; i < STEPS; ++i) {
        // call update
        HANDLE_ERROR(cudaEventRecord(start_update, 0));
        aosoa_update_with_shared<<<(PROBLEMSIZE + LANES - 1) / LANES, LANES>>>(particle_block_device);
        HANDLE_LAST_ERROR;
        HANDLE_ERROR(cudaEventRecord(stop_update, 0));
        // call move
        HANDLE_ERROR(cudaEventRecord(start_move, 0));
        aosoa_move<<<(PROBLEMSIZE + LANES - 1) / LANES, LANES>>>(particle_block_device);
        HANDLE_LAST_ERROR;
        HANDLE_ERROR(cudaEventRecord(stop_move));

        HANDLE_ERROR(cudaEventSynchronize(stop_move));
        HANDLE_ERROR(cudaEventElapsedTime(&time_update, start_update, stop_update));
        HANDLE_ERROR(cudaEventElapsedTime(&time_move, start_move, stop_move));
        printf("AoSoA\t%3.4fms\t%3.4fms\n", time_update, time_move);
        sum_move += time_move;
        sum_update += time_update;
    }
    printf("AVG:\t%3.4fms\t%3.6fms\n\n", sum_update / STEPS,  sum_move / STEPS);

    // maybe write back

    // free mem
    HANDLE_ERROR(cudaFree(particle_block_device));
    HANDLE_ERROR(cudaDeviceReset());
}
