#include <stdio.h>
#include <math.h>

__global__ void kernel(float *a, int offset)
{
	int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
	float x = (float)i;
	float s = sinf(x);
	float c = cosf(x);
	a[i] += sqrtf(s*s + c*c);
}

float maxError(float *a, int n)
{
	float maxE = 0;
	for (int i = 0; i < n; i++)
	{
		float error = fabs(a[i]-1.0f);
		if (error > maxE) maxE = error;
	}
	return maxE;
}

int main(int argc, char **argv)
{
	const int blockSize = 256, nStreams = 4;
	const int n = 256 * 1024 * blockSize * nStreams;
	const int streamSize = n / nStreams; // How many go into each stream
	const int streamBytes = streamSize * sizeof(float); // Num bytes for each stream
	const int bytes = n * sizeof(float); // Total num bytes for all n


	// Allocate pinned host memory and device memory

	float *a, *d_a;
	cudaMallocHost((void**)&a, bytes); // On the host
	cudaMalloc((void**)&d_a, bytes); // On the device

	float ms;
	cudaEvent_t startEvent, stopEvent, dummyEvent;
	cudaStream_t stream[nStreams]; // holds all the streams
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventCreate(&dummyEvent);

	for (int i = 0; i < nStreams; i++)
	{
		cudaStreamCreate(&stream[i]); // Makes all the streams
	}

	// Baseline case - everything follows in sequence
	
	memset(a, 0, bytes);
	cudaEventRecord(startEvent, 0);
	cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
	

	// kernel<<<#THREADS IN A BLOCK, SIZE OF BLOCK>>>(...)
	kernel<<<n/blockSize, blockSize>>>(d_a, 0);
	cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&ms, startEvent, stopEvent);

	printf("Time for sequential (baseline) transfer and execute: %f ms\n", ms);
	printf("Max error: %e\n", maxError(a, n));


	// Asynchronous version 1: loop over {copy to d, kernel, copy to h}

	memset(a, 0, bytes);
	cudaEventRecord(startEvent, 0);
	for (int i = 0; i < nStreams; ++i) // So for each stream individually
	{
		int offset = i * streamSize; // Ex: stream 2 starts where stream 2 ends
		cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
		
		kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);

		cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
	}	
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&ms, startEvent, stopEvent);
	
	printf("Time for aysnc V1 transfer and execute: %f ms\n", ms);
	printf("Max error: %e\n", maxError(a, n));
	
	// Asynchronous version 2: loop over all copy, all kernel, all copy

	memset(a, 0, bytes);
	cudaEventRecord(startEvent, 0);
	for (int i = 0; i < nStreams; ++i)
	{
		int offset = i * streamSize;
		cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
	}

	for (int i = 0; i < nStreams; ++i)
	{
		int offset = i * streamSize;
		kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
	}
	
	for (int i = 0; i < nStreams; ++i)
	{
		int offset = i * streamSize;
		cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
	}

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&ms, startEvent, stopEvent);

	printf("Time for async V2 transfer and execute: %f ms\n", ms);
	printf("Max error: %e\n", maxError(a, n));


	cudaFree(d_a);
	cudaFreeHost(a);
}
