#include "array_of_structs.cu"
#include "struct_of_arrays.cu"
#include <stdio.h>

int main(void) {
    printf("%d k particles (%f kiB)\n", kProblemSize,
           (float) (kProblemSize * sizeof(float) * 7 / 1024));
    print_cuda_infos_at_start();
    // aos_run();
    soa_run();
    // asosa_run();

    return 0;
}