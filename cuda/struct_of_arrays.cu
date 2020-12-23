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
    x_values[other] = 0;
    y_values[other] = 0;
    z_values[other] = 0;
    float posix = posx[id];
    float posiy = posy[id];
    float posiz = posz[id];

    for (std::size_t j = other; j < PROBLEMSIZE; j += 1024) {
        const float xdistance = posix - posx[j];
        const float ydistance = posiy - posy[j];
        const float zdistance = posiz - posz[j];
        const float xdistanceSqr = xdistance * xdistance;
        const float ydistanceSqr = ydistance * ydistance;
        const float zdistanceSqr = zdistance * zdistance;
        const float distSqr = EPS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
        const float distSixth = distSqr * distSqr * distSqr;
        const float invDistCube = 1.0f / std::sqrt(distSixth);
        const float sts = mass[j] * invDistCube * TIMESTEP;
        x_values[other] += xdistanceSqr * sts;
        y_values[other] += ydistanceSqr * sts;
        z_values[other] += zdistanceSqr * sts;
    }
    // reduce to one
    // Todo improve with half steps
    SYNC_THREADS;
    if (id == 0) {
        for (int j = 1; j < 1024; j++) {

            x_values[0] += x_values[j];
            y_values[0] += y_values[j];
            z_values[0] += z_values[j];
        }
        velx[id] += x_values[0];
        vely[id] += y_values[0];
        velz[id] += z_values[0];
    }
    SYNC_THREADS;
}

__global__ void soa_move(float *posx, float *posy, float *posz, float *velx,
                         float *vely, float *velz) {
    int id = LINEAR_ID;
    if ( id < PROBLEMSIZE) {
        posx[id] += velx[id] * TIMESTEP;
        posy[id] += vely[id] * TIMESTEP;
        posz[id] += velz[id] * TIMESTEP;
    }
}


// Todo create struct to simplify code

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

    for (std::size_t s = 0; s < STEPS; ++s) {
        
        HANDLE_ERROR(cudaEventRecord(start, 0));
        soa_update<<<PROBLEMSIZE, 1024>>>(posx_d, posy_d, posz_d, velx_d, vely_d, velz_d, mass_d);
        cudaEventRecord(end, 0);

        cudaEventRecord(start2, 0);
        soa_move<<<(PROBLEMSIZE+1023) / 1024, 1024>>>(posx_d, posy_d, posz_d, velx_d, vely_d, velz_d);
        cudaEventRecord(end2, 0);
        HANDLE_ERROR(cudaEventSynchronize(end2));
       
        float time;
        cudaEventElapsedTime(&time, start, end);
        float time2;
        cudaEventElapsedTime(&time2, start2, end2);
        std::cout << "SoA\t" << time << "ms" << '\t' << time2 << "ms" << '\n';
    }
    HANDLE_ERROR(cudaMemcpy(posx_h, posx_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(posy_h, posy_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(posz_h, posz_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(velx_h, velx_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(vely_h, vely_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(velz_h, velz_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mass_h, mass_d, PROBLEMSIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free mem
    HANDLE_ERROR(cudaFree(posx_d));
    HANDLE_ERROR(cudaFree(posy_d));
    HANDLE_ERROR(cudaFree(posz_d));
    HANDLE_ERROR(cudaFree(velx_d));
    HANDLE_ERROR(cudaFree(vely_d));
    HANDLE_ERROR(cudaFree(velz_d));
    HANDLE_ERROR(cudaFree(mass_d));
    // maybe reset
    // cudaDeviceReset();
}