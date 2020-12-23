#include "shared_header.h"

struct particle_block        // deleted alignas(64)
{
    struct
    {
        float x[LANES];
        float y[LANES];
        float z[LANES];
    } pos;
    struct
    {
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
        float* pivelx,
        float* pively,
        float* pivelz,
        float pjposx,
        float pjposy,
        float pjposz,
        float pjmass)
        {
    float xdistance = piposx - pjposx;
    float ydistance = piposy - pjposy;
    float zdistance = piposz - pjposz;
    xdistance *= xdistance;
    ydistance *= ydistance;
    zdistance *= zdistance;
    const float distSqr = EPS2 + xdistance + ydistance + zdistance;
    const float distSixth = distSqr * distSqr * distSqr;
    const float invDistCube = 1.0f / std::sqrt(distSixth);
    const float sts = pjmass * invDistCube * TIMESTEP;
    *pivelx += xdistance * sts;
    *pively += ydistance * sts;
    *pivelz += zdistance * sts;
}

__global__ void aosoa_update(particle_block* particles){
    particle_block mainBlock = particles[blockIdx.x];
    int mainLane = threadIdx.x;
    for (int otherBlockIndex = 0; otherBlockIndex < BLOCKS; ++otherBlockIndex){
        for (int otherLane = 0; otherLane < LANES; ++otherLane){
            particle_block otherBlock = particles[otherBlockIndex];
        pPInteraction( mainBlock.pos.x[mainLane],
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


__global__ void aosoa_move(struct particle_block* particle_block){
    int bi = blockIdx.x;
    int i = threadIdx.x;

    struct particle_block block = particle_block[bi];
    block.pos.x[i] += block.vel.x[i] * TIMESTEP;
    block.pos.y[i] += block.vel.y[i] * TIMESTEP;
    block.pos.z[i] += block.vel.z[i] * TIMESTEP;
}



void aosoa_run(){

    // init event management
    cudaEvent_t start_update, stop_update;
    cudaEventCreate(&start_update);
    cudaEventCreate(&stop_update);

    cudaEvent_t start_move, stop_move;
    cudaEventCreate(&start_move);
    cudaEventCreate(&stop_move);

    // "allocate" mem
    struct particle_block particle_block_host[BLOCKS];
    struct particle_block* particle_block_device;

    // fill with random values
    // iterate over the structs 'stru' in the array
    for (int stru = 0; stru < ( sizeof(particle_block_host) / sizeof(struct particle_block) ); ++stru) {
        // iterate over pos-array inside the struct, then over the vel-arrays, then mass-array
        for (int l = 0; l < LANES; ++l) {
            particle_block_host[stru].pos.x[l] = (float)rand();
            particle_block_host[stru].pos.y[l] = (float)rand();
            particle_block_host[stru].pos.z[l] = (float)rand();

            particle_block_host[stru].vel.x[l] = (float)rand() / 10.0f;
            particle_block_host[stru].vel.y[l] = (float)rand() / 10.0f;
            particle_block_host[stru].vel.z[l] = (float)rand() / 10.0f;

            particle_block_host[stru].mass[l] = (float)rand() / 100.0f;
        }
    }

    int datasize = sizeof(particle_block_host);
    HANDLE_ERROR( cudaMalloc(&particle_block_device, datasize) );
    HANDLE_ERROR( cudaMemcpy(particle_block_device, particle_block_host, datasize, cudaMemcpyHostToDevice) );


    float time_update, time_move;
    for (int i=0; i< STEPS; ++i) {
        // call update
        cudaEventRecord(start_update, 0);
        aosoa_update<<<(PROBLEMSIZE +LANES-1)/LANES, LANES>>>(particle_block_device);
        HANDLE_LAST_ERROR;
        cudaEventRecord(stop_update, 0);
        // call move
        cudaEventRecord(start_move, 0);
        aosoa_move<<<(PROBLEMSIZE +LANES-1)/LANES, LANES>>>(particle_block_device);
        HANDLE_LAST_ERROR;
        cudaEventRecord(stop_move, 0);

        cudaEventSynchronize(stop_move);
        cudaEventElapsedTime(&time_update, start_update, stop_update);
        printf("AoSoA\t%fms\t%fms\n", time_update, time_move);
        cudaEventElapsedTime(&time_move, start_move, stop_move);
    }

    // maybe write back

    // free mem
    HANDLE_ERROR( cudaFree(particle_block_device) );
    HANDLE_ERROR( cudaDeviceReset() );
}
