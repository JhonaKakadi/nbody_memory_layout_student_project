#include "shared_header.h"

__global__ void soa_update(float *posx, float *posy, float *posz, float *velx, float *vely, float *velz, float *mass) {
    // one kernel per particle
    // 1024 threads per particle
    // Todo use constant mem for save original points
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
        const float xdistance = posix - posx[j];
        const float ydistance = posiy - posy[j];
        const float zdistance = posiz - posz[j];
        const float xdistanceSqr = xdistance * xdistance;
        const float ydistanceSqr = ydistance * ydistance;
        const float zdistanceSqr = zdistance * zdistance;
        const float distSqr = EPS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
        const float distSixth = distSqr * distSqr * distSqr;
        const float invDistCube = 1.0f / sqrt(distSixth);
        const float sts = mass[j] * invDistCube * TIMESTEP;
        x_run += xdistanceSqr * sts;
        y_run += ydistanceSqr * sts;
        z_run += zdistanceSqr * sts;
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
    srand(NULL);
    for (std::size_t i = 0; i < PROBLEMSIZE; ++i) {
        posx_h[i] = (float) rand();
        posy_h[i] = (float) rand();
        posz_h[i] = (float) rand();
        velx_h[i] = (float) rand() / 10.0f;
        vely_h[i] = (float) rand() / 10.0f;
        velz_h[i] = (float) rand() / 10.0f;
        mass_h[i] = (float) rand() / 100.0f;
    }

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
    // copy points to Device
    HANDLE_ERROR(cudaMemcpy(posx_d, posx_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(posy_d, posy_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(posz_d, posz_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velx_d, velx_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(vely_d, vely_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velz_d, velz_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mass_d, mass_h, PROBLEMSIZE * sizeof(float), cudaMemcpyHostToDevice));

    //
    cudaEvent_t start, end;
    cudaEvent_t start2, end2;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));
    HANDLE_ERROR(cudaEventCreate(&start2));
    HANDLE_ERROR(cudaEventCreate(&end2));

    float sum_move = 0, sum_update = 0;
    for (std::size_t s = 0; s < STEPS; ++s) {

        HANDLE_ERROR(cudaEventRecord(start, 0));
        soa_update<<<PROBLEMSIZE, 1024>>>(posx_d, posy_d, posz_d, velx_d, vely_d, velz_d, mass_d);
        cudaEventRecord(end, 0);

        cudaEventRecord(start2, 0);
        soa_move<<<(PROBLEMSIZE + 1023) / 1024, 1024>>>(posx_d, posy_d, posz_d, velx_d, vely_d, velz_d);
        HANDLE_ERROR(cudaEventRecord(end2, 0));
        HANDLE_ERROR(cudaEventSynchronize(end2));

        float time_update;
        HANDLE_ERROR(cudaEventElapsedTime(&time_update, start, end));
        float time_move;
        HANDLE_ERROR(cudaEventElapsedTime(&time_move, start2, end2));
        std::cout << "SoA\t" << time_update << "ms" << '\t' << time_move << "ms" << '\n';
        sum_move += time_move;
        sum_update += time_update;
    }
    printf("AVG:\t%3.4fms\t%3.6fms\n\n", sum_update / STEPS, sum_move / STEPS);

    // copy back
    HANDLE_ERROR(cudaMemcpy(posx_h, posx_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(posy_h, posy_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(posz_h, posz_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(velx_h, velx_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(vely_h, vely_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(velz_h, velz_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mass_h, mass_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // destroy events
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(start2));
    HANDLE_ERROR(cudaEventDestroy(end));
    HANDLE_ERROR(cudaEventDestroy(end2));

    // Free mem
    HANDLE_ERROR(cudaFree(posx_d));
    HANDLE_ERROR(cudaFree(posy_d));
    HANDLE_ERROR(cudaFree(posz_d));
    HANDLE_ERROR(cudaFree(velx_d));
    HANDLE_ERROR(cudaFree(vely_d));
    HANDLE_ERROR(cudaFree(velz_d));
    HANDLE_ERROR(cudaFree(mass_d));

    HANDLE_ERROR(cudaDeviceReset());
}