import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np


# CUDA code that should add 1 to all elements of the array *a

code = """
	#include <stdio.h>
	#include <math.h>
	__global__ void kernel(float *a, int offset)
	{		
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		float x = (float)i;
		float s = sinf(x);
		float c = cosf(x);
		a[i] += sqrtf(s*s + c*c);
	}
	"""
mod = SourceModule(code)

kernel = mod.get_function("kernel")


# Useful variables for how to allocate work to each stream

blockSize = 1024
nStreams = 2
n = 2 * 1024 * blockSize * nStreams
streamSize = n / nStreams;
streamBytes = streamSize * (32 / 8) # np.float32
bytes = n * 4 # 32/8

# init array A on both host and device
a = []
a_gpu = []

# init STREAMS as array of streams, A and A_GPU arrays where ith element is the partion of the data that streams[i] is assigned to 
streams, events = [], []
for i in range(nStreams):
	streams.append(cuda.Stream())
	a.append(np.zeros(streamSize, dtype=np.float32))
	a_gpu.append(cuda.mem_alloc(a[i].nbytes))
	events.append(dict([("start", cuda.Event()), ("end", cuda.Event())]))

for i in range(nStreams):
	cuda.memcpy_htod(a_gpu[i], a[i])


for i in range(nStreams):
	offset = np.int32(i * streamSize)
	print offset
	#cuda.memcpy_htod(a_gpu[i], a[i])
	events[i]["start"].record(streams[i])
	kernel(a_gpu[i], offset, block=(blockSize, 1, 1), stream=streams[i])
	
for i in range(nStreams):
	events[i]["end"].record(streams[i])


for i in range(nStreams):
	cuda.memcpy_dtoh(a[i], a_gpu[i])

a = np.asarray(a)
a = a.flatten()[1]
print a
print np.where(a == 0)
