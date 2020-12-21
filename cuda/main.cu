#include <stdio.h>
#include "shared_header.h"

#include "array_of_structs.cu"
#include "struct_of_arrays.cu"

int main(void) {
	printf("%d k particles (%f kiB)\n", kProblemSize, (float)(kProblemSize * sizeof(float) * 7 / 1024) );

    aos_run();
    soa_run();
    asosa_run();

    return 0;
}