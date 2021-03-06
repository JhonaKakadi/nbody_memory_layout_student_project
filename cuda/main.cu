#include "array_of_structs.cu"
#include "struct_of_arrays.cu"
#include "array_of_struct_of_arrays.cu"
#include <stdio.h>

int main(void) {

    print_cuda_infos_at_start();
	printf("%d k particles (%.3f kiB)\n", PROBLEMSIZE / 1000,
		(float)(PROBLEMSIZE * sizeof(float) * 7 / 1024) );
   
    aos_run();
    soa_run();
    aosoa_run();

    return 0;
}