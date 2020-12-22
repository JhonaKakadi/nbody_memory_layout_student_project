#include "shared_header.h"


__global__ void soa_update(float* posx, float* posy, float* posz, float* velx, float* vely, float* velz, float* mass) {
    // one kernel per particle
    // 1024 threads per particle
    // Todo use constant mem for save original points
    // Note 32 is size of warp
    int id = LINEAR_ID;
    int other = 0;
    for (std::size_t j = id; j < PROBLEMSIZE; j += 1024) {
        const float xdistance = posx[id] - posx[other];
        const float ydistance = posy[id] - posy[other];
        const float zdistance = posz[id] - posz[other];
        const float xdistanceSqr = xdistance * xdistance;
        const float ydistanceSqr = ydistance * ydistance;
        const float zdistanceSqr = zdistance * zdistance;
        const float distSqr = EPS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
        const float distSixth = distSqr * distSqr * distSqr;
        const float invDistCube = 1.0f / std::sqrt(distSixth);
        const float sts = mass[other] * invDistCube * TIMESTEP;
        velx[id] += xdistanceSqr * sts;
        vely[id] += xdistanceSqr * sts;
        velz[id] += xdistanceSqr * sts;
    }
}
/*
__global__ void move() {
  int id = LINEAR_ID;
  for (std::size_t i = id; i < kProblemSize; i += 1024) {
    posx[i] += velx[i] * kTimestep;
    posy[i] += vely[i] * kTimestep;
    posz[i] += velz[i] * kTimestep;
  }
}*/

void soa_run() {

    float* posx_h = (float*)malloc(sizeof(float) * kProblemSize);
    float* posy_h = (float*)malloc(sizeof(float) * kProblemSize);
    float* posz_h = (float*)malloc(sizeof(float) * kProblemSize);
    float* velx_h = (float*)malloc(sizeof(float) * kProblemSize);
    float* vely_h = (float*)malloc(sizeof(float) * kProblemSize);
    float* velz_h = (float*)malloc(sizeof(float) * kProblemSize);
    float* mass_h = (float*)malloc(sizeof(float) * kProblemSize);
    srand(NULL);
    for (std::size_t i = 0; i < kProblemSize; ++i) {
        posx_h[i] = (float)rand();
        posy_h[i] = (float)rand();
        posz_h[i] = (float)rand();
        velx_h[i] = (float)rand() / 10.0f;
        vely_h[i] = (float)rand() / 10.0f;
        velz_h[i] = (float)rand() / 10.0f;
        mass_h[i] = (float)rand() / 100.0f;
    }

    float* posx_d;
    float* posy_d;
    float* posz_d;
    float* velx_d;
    float* vely_d;
    float* velz_d;
    float* mass_d;

    cudaMalloc(&posx_d, kProblemSize * sizeof(float));
    cudaMalloc(&posy_d, kProblemSize * sizeof(float));
    cudaMalloc(&posz_d, kProblemSize * sizeof(float));
    cudaMalloc(&velx_d, kProblemSize * sizeof(float));
    cudaMalloc(&vely_d, kProblemSize * sizeof(float));
    cudaMalloc(&velz_d, kProblemSize * sizeof(float));
    cudaMalloc(&mass_d, kProblemSize * sizeof(float));
    // copy points to Device
    cudaMemcpy(posx_d, posx_h, kProblemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(posy_d, posy_h, kProblemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(posz_d, posz_h, kProblemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(velx_d, velx_h, kProblemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(vely_d, vely_h, kProblemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(velz_d, velz_h, kProblemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(mass_d, mass_h, kProblemSize, cudaMemcpyHostToDevice);

    //
    cudaEvent_t start, end;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));

    // for (std::size_t s = 0; s <  kSteps; ++s) {
    cudaEventRecord(start, 0);
    std::cout << "Kernel\n";
    soa_update << <kProblemSize, 1024 >> > (posx_d, posy_d, posz_d, velx_d, vely_d, velz_d, mass_d);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);

    // move<<<kProblemSize / 1024, 1024>>>(posx.data(), posy.data(), posz.data(),
    //                                    velx.data(), vely.data(), velz.data());
    // sumMove += watch.elapsedAndReset();
    // }
    std::cout << "SoA\t" << time / kSteps << "ms" << '\t' << '\n';
}