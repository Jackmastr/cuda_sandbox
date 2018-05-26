import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

code = """
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
	"""
mod = SourceModule(code)

blockSize = 256
nStreams = 4
n = 256 * 1024 * blockSize * nStreams
streamSize = n / nStreams;
streamBytes = streamSize * (32 / 8) # np.float32
bytes = n * 4 # 32/8

a_gpu = gpuarray.to_gpu(np.zeros(n, dtype=np.float32))

streams = np.array(nStreams)
for i in range(nStreams):
	streams.append(cuda.Stream())

kernel = mod.get_function("kernel")

for i in range(nStreams):
	offset = i * streamSize
	a_gpu #hmmmmm
