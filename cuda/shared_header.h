// contains global constants

const int kProblemSize = 16 * 1024;
const int kSteps = 5;
const float kTimestep = 0.0001f;
const float kEPS2 = 0.01f;


// contains definition of error handler

#define HANDLE_ERROR(err)\
	(handleCudaError(err, __FILE__, __LINE__))

static void handleCudaError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("[%s] in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}