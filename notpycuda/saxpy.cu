#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride)
	{
		y[i] = a*x[i] + y[i];
	}
}


int main(void)
{
	int N = 1<<20;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));

	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpyKind hTD = cudaMemcpyHostToDevice;

	cudaMemcpy(d_x, x, N*sizeof(float), hTD);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1 Million Elements
	cudaEventRecord(start);
	saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
	cudaEventRecord(stop);

	// Error Checking
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("time to complete: %f ms.\n", milliseconds);

	// Testing
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		maxError = max(maxError, abs(y[i]-4.0f));
	}
	printf("Max error: %f\n", maxError);
	printf("Effective Bandwidth (GB/s): %f\n", (N*4*3)/(milliseconds*1e6));

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}
