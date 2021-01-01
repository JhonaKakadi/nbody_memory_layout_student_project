#include "shared_header.h"


__device__ inline void
soa_pp_Interaction(const float posix, const float posiy, const float posiz,
                   float pos_x_other, float pos_y_other, float pos_z_other, float mass_other,
                   float *x_run, float *y_run, float *z_run) {
    const float xdistance = posix - pos_x_other;
    const float ydistance = posiy - pos_y_other;
    const float zdistance = posiz - pos_z_other;
    const float xdistanceSqr = xdistance * xdistance;
    const float ydistanceSqr = ydistance * ydistance;
    const float zdistanceSqr = zdistance * zdistance;
    const float distSqr = EPS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
    const float distSixth = distSqr * distSqr * distSqr;
    // TODO Use sqrtf() with correct flags
    const float invDistCube = 1.0f / sqrt(distSixth);
    const float sts = mass_other * invDistCube * TIMESTEP;
    *x_run += xdistanceSqr * sts;
    *y_run += ydistanceSqr * sts;
    *z_run += zdistanceSqr * sts;
}


__global__ void
soa_update_k(float* posx, float* posy, float* posz, float* velx, float* vely, float* velz, float* mass) {
    // one kernel per particle
    // 1024 threads per particle
    // Todo choose good names for vars
    // Note 32 is size of warp
    int id = blockIdx.x;
    int other = threadIdx.x;

    float x_run = 0;
    float y_run = 0;
    float z_run = 0;
    const float posix = posx[id];
    const float posiy = posy[id];
    const float posiz = posz[id];

    for (std::size_t j = other; j < PROBLEMSIZE; j += 1024) {
        soa_pp_Interaction(posix, posiy, posiz, posx[j], posy[j], posz[j], mass[j], &x_run, &y_run, &z_run);
    }
    atomicAdd(&velx[id], x_run);
    atomicAdd(&vely[id], y_run);
    atomicAdd(&velz[id], z_run);    
}



__global__ void
soa_update_k_shared(float *posx, float *posy, float *posz, float *velx, float *vely, float *velz, float *mass) {
    // one kernel per particle
    // 1024 threads per particle
    // Todo choose good names for vars
    // Note 32 is size of warp
    int id = blockIdx.x;
    int other = threadIdx.x;
    __shared__ float x_values[1024];
    __shared__ float y_values[1024];
    __shared__ float z_values[1024];

    float x_run = 0;
    float y_run = 0;
    float z_run = 0;
    const float posix = posx[id];
    const float posiy = posy[id];
    const float posiz = posz[id];


    for (std::size_t j = other; j < PROBLEMSIZE; j += 1024) {
        soa_pp_Interaction(posix, posiy, posiz, posx[j], posy[j], posz[j], mass[j], &x_run, &y_run, &z_run);
    }
    x_values[other] = x_run;
    y_values[other] = y_run;
    z_values[other] = z_run;

    // reduce to one
    SYNC_THREADS;
    int border = 512;
    while (border != 0) {
        if (other < border) {
            x_values[other] += x_values[other + border];
            y_values[other] += y_values[other + border];
            z_values[other] += z_values[other + border];
        }
        border /= 2;
        SYNC_THREADS;
    }
    // write back to global mem
    if (other == 0) {
        velx[id] += x_values[0];
        vely[id] += y_values[0];
        velz[id] += z_values[0];
    }
}





__global__ void
soa_update_t(float* posx, float* posy, float* posz, float* velx, float* vely, float* velz, float* mass) {
    // one kernel for 1024 particles
    // 1 thread per particle
    // Todo choose good names for vars
    // Note 32 is size of warp
    int id = LINEAR_ID;

    float x_run = 0;
    float y_run = 0;
    float z_run = 0;
    const float posix = posx[id];
    const float posiy = posy[id];
    const float posiz = posz[id];

    int max_iter = 1024;
    for (std::size_t other = 0; other < PROBLEMSIZE; ++other) {
        if (id < PROBLEMSIZE) {
            soa_pp_Interaction(posix, posiy, posiz, posx[other], posy[other], posz[other], mass[other], &x_run, &y_run, &z_run);
        }
    }
    velx[id] += x_run;
    vely[id] += y_run;
    velz[id] += z_run;
}



__global__ void
soa_update_t_shared(float *posx, float *posy, float *posz, float *velx, float *vely, float *velz, float *mass) {
    // one kernel per 1024 particles
    // 1 thread per particle
    // Todo choose good names for vars
    // Note 32 is size of warp
    int id = LINEAR_ID;

    float x_run = 0;
    float y_run = 0;
    float z_run = 0;
    const float posix = posx[id];
    const float posiy = posy[id];
    const float posiz = posz[id];
    __shared__ float temp_x[1024];
    __shared__ float temp_y[1024];
    __shared__ float temp_z[1024];
    __shared__ float temp_mass[1024];
    int max_iter = 1024;
    for (std::size_t j = 0; j < (PROBLEMSIZE + 1023) / 1024; ++j) {
        if (threadIdx.x + j * 1024 < PROBLEMSIZE) {
            temp_x[threadIdx.x] = posx[j * 1024 + threadIdx.x];
            temp_y[threadIdx.x] = posy[j * 1024 + threadIdx.x];
            temp_z[threadIdx.x] = posz[j * 1024 + threadIdx.x];
            temp_mass[threadIdx.x] = mass[j * 1024 + threadIdx.x];
        }
        SYNC_THREADS;
        if (id < PROBLEMSIZE) {
            if (PROBLEMSIZE - (j * 1024) < 1024) {
                max_iter = PROBLEMSIZE % 1024;
            }
            for (int index = 0; index < max_iter; ++index) {
                soa_pp_Interaction(posix, posiy, posiz, temp_x[index], temp_y[index], temp_z[index], temp_mass[index], &x_run, &y_run, &z_run);
            }
        }
        //necessary?
        SYNC_THREADS;
    }
    velx[id] += x_run;
    vely[id] += y_run;
    velz[id] += z_run;
}

__global__ void soa_randNormal(float* posx, float* posy, float* posz, float* velx, float* vely, float* velz, float* mass) {
    int tid = LINEAR_ID;
    curandState state;
    if (tid < PROBLEMSIZE) {
        curand_init(1337, tid, 0, &state);

        // write random values to particle attributes
        posx[tid] = curand_normal(&state);
        velx[tid] = curand_normal(&state) / 10.0f;
        posy[tid] = curand_normal(&state);
        vely[tid] = curand_normal(&state) / 10.0f;
        posz[tid] = curand_normal(&state);
        velz[tid] = curand_normal(&state) / 10.0f;

        mass[tid] = curand_normal(&state) / 100.0f;
    }
}



__global__ void soa_move(float *posx, float *posy, float *posz, float *velx,
                         float *vely, float *velz) {
    int id = LINEAR_ID;
    if (id < PROBLEMSIZE) {
        posx[id] += velx[id] * TIMESTEP;
        posy[id] += vely[id] * TIMESTEP;
        posz[id] += velz[id] * TIMESTEP;
    }
}


// Todo create struct to simplify code

struct particles {
    float x_pos[PROBLEMSIZE];
    float y_pos[PROBLEMSIZE];
    float z_pos[PROBLEMSIZE];
    float x_vel[PROBLEMSIZE];
    float y_vel[PROBLEMSIZE];
    float z_vel[PROBLEMSIZE];
    // Todo map to texture
    float mass[PROBLEMSIZE];
};


void soa_run() {

    float *posx_h = (float *) malloc(sizeof(float) * PROBLEMSIZE);
    float *posy_h = (float *) malloc(sizeof(float) * PROBLEMSIZE);
    float *posz_h = (float *) malloc(sizeof(float) * PROBLEMSIZE);
    float *velx_h = (float *) malloc(sizeof(float) * PROBLEMSIZE);
    float *vely_h = (float *) malloc(sizeof(float) * PROBLEMSIZE);
    float *velz_h = (float *) malloc(sizeof(float) * PROBLEMSIZE);
    float *mass_h = (float *) malloc(sizeof(float) * PROBLEMSIZE);
    
   

    float *posx_d;
    float *posy_d;
    float *posz_d;
    float *velx_d;
    float *vely_d;
    float *velz_d;
    float *mass_d;


    HANDLE_ERROR(cudaMalloc(&posx_d, PROBLEMSIZE * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&posy_d, PROBLEMSIZE * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&posz_d, PROBLEMSIZE * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&velx_d, PROBLEMSIZE * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&vely_d, PROBLEMSIZE * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&velz_d, PROBLEMSIZE * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&mass_d, PROBLEMSIZE * sizeof(float)));

    soa_randNormal<<<(PROBLEMSIZE+1023)/1024,1024>>>(posx_d,posy_d,posz_d,velx_d,vely_d,velz_d,mass_d);

    // copy points to Device
    /*
    HANDLE_ERROR(cudaMemcpy(posx_d, posx_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(posy_d, posy_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(posz_d, posz_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velx_d, velx_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(vely_d, vely_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velz_d, velz_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mass_d, mass_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    */
    //
    cudaEvent_t start_update_k, end_update_k;
    cudaEvent_t start_update_k_shared, end_update_k_shared;
    cudaEvent_t start_update_t, end_update_t;
    cudaEvent_t start_update_t_shared, end_update_t_shared;
    cudaEvent_t start_move, end_move;

    HANDLE_ERROR(cudaEventCreate(&start_update_k)); 
    HANDLE_ERROR(cudaEventCreate(&end_update_k));
    
    HANDLE_ERROR(cudaEventCreate(&start_update_k_shared));
    HANDLE_ERROR(cudaEventCreate(&end_update_k_shared));
    
    HANDLE_ERROR(cudaEventCreate(&start_update_t));
    HANDLE_ERROR(cudaEventCreate(&end_update_t));

    HANDLE_ERROR(cudaEventCreate(&start_update_t_shared));
    HANDLE_ERROR(cudaEventCreate(&end_update_t_shared));
    
    HANDLE_ERROR(cudaEventCreate(&start_move));
    HANDLE_ERROR(cudaEventCreate(&end_move));

    float sum_move = 0, sum_update_k = 0, sum_update_t = 0;
    float sum_update_k_shared = 0, sum_update_t_shared;
    float time_move;
    float time_update_k;
    float time_update_k_shared;
    float time_update_t;
    float time_update_t_shared;

    printf("Benchmarks: Kernel, \tKernel_shared, \tThread, \tThread_shared, \tmove\n");
    for (std::size_t s = 0; s < STEPS; ++s) {

        HANDLE_ERROR(cudaEventRecord(start_update_k, 0));
        soa_update_k << <PROBLEMSIZE, 1024 >> > (posx_d, posy_d, posz_d, velx_d, vely_d, velz_d, mass_d);
        HANDLE_ERROR(cudaEventRecord(end_update_k, 0));

        HANDLE_ERROR(cudaEventRecord(start_update_k_shared, 0));
        soa_update_k_shared <<<PROBLEMSIZE, 1024>>>(posx_d, posy_d, posz_d, velx_d, vely_d, velz_d, mass_d);
        HANDLE_ERROR(cudaEventRecord(end_update_k_shared, 0));

        
        HANDLE_ERROR(cudaEventRecord(start_update_t, 0));
        soa_update_t <<<(PROBLEMSIZE + 1023) / 1024, 1024 >>>(posx_d, posy_d, posz_d, velx_d, vely_d, velz_d,
                                                                     mass_d);
        HANDLE_ERROR(cudaEventRecord(end_update_t, 0));
        
        HANDLE_ERROR(cudaEventRecord(start_update_t_shared, 0));
        soa_update_t_shared <<<(PROBLEMSIZE + 1023) / 1024, 1024 >>>(posx_d, posy_d, posz_d, velx_d, vely_d, velz_d,
                                                                     mass_d);
        HANDLE_ERROR(cudaEventRecord(end_update_t_shared, 0));

        HANDLE_ERROR(cudaEventRecord(start_move, 0));
        soa_move <<<(PROBLEMSIZE + 1023) / 1024, 1024>>>(posx_d, posy_d, posz_d, velx_d, vely_d, velz_d);
        HANDLE_ERROR(cudaEventRecord(end_move, 0));

        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaEventElapsedTime(&time_update_k, start_update_k, end_update_k));
        HANDLE_ERROR(cudaEventElapsedTime(&time_update_k_shared, start_update_k_shared, end_update_k_shared));
        HANDLE_ERROR(cudaEventElapsedTime(&time_update_t, start_update_t, end_update_t));
        HANDLE_ERROR(cudaEventElapsedTime(&time_update_t_shared, start_update_t_shared, end_update_t_shared));
        HANDLE_ERROR(cudaEventElapsedTime(&time_move, start_move, end_move));
        printf("SoA\t%3.4fms\t%3.4fms\t%3.4fms\t%3.4fms\t%3.4fms\n", time_update_k, time_update_k_shared, time_update_t, time_update_t_shared, time_move);
        sum_move += time_move;
        sum_update_k += time_update_k;
        sum_update_k_shared += time_update_k_shared;
        sum_update_t += time_update_t;
        sum_update_t_shared += time_update_t_shared;
    }
    printf("AVG:\t%3.4fms\t%3.4fms\t%3.4fms\t%3.4fms\t%3.6fms\n\n", sum_update_k / STEPS, sum_update_k_shared / STEPS, sum_update_t/STEPS, sum_update_t_shared/STEPS, sum_move / STEPS);

    // copy back
    HANDLE_ERROR(cudaMemcpy(posx_h, posx_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(posy_h, posy_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(posz_h, posz_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(velx_h, velx_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(vely_h, vely_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(velz_h, velz_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mass_h, mass_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // destroy events
    HANDLE_ERROR(cudaEventDestroy(start_update_k));
    HANDLE_ERROR(cudaEventDestroy(start_update_k_shared));
    HANDLE_ERROR(cudaEventDestroy(start_update_t));
    HANDLE_ERROR(cudaEventDestroy(start_update_t_shared));
    HANDLE_ERROR(cudaEventDestroy(start_move));
    HANDLE_ERROR(cudaEventDestroy(end_update_k));
    HANDLE_ERROR(cudaEventDestroy(end_update_k_shared));
    HANDLE_ERROR(cudaEventDestroy(end_update_t)); 
    HANDLE_ERROR(cudaEventDestroy(end_update_t_shared));
    HANDLE_ERROR(cudaEventDestroy(end_move));

    // Free mem device
    HANDLE_ERROR(cudaFree(posx_d));
    HANDLE_ERROR(cudaFree(posy_d));
    HANDLE_ERROR(cudaFree(posz_d));
    HANDLE_ERROR(cudaFree(velx_d));
    HANDLE_ERROR(cudaFree(vely_d));
    HANDLE_ERROR(cudaFree(velz_d));
    HANDLE_ERROR(cudaFree(mass_d));
    // Free mem host
    free(posx_h);
    free(posy_h);
    free(posz_h);
    free(velx_h);
    free(vely_h);
    free(velz_h);
    free(mass_h);

    HANDLE_ERROR(cudaDeviceReset());
}