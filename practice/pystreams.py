#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np


# CUDA code that should add 1 to all elements of the array *a

code = """
	#include <stdio.h>
	#include <math.h>
	__global__ void kernel(float *a, int n)
	{		
		int i = threadIdx.x + blockIdx.x * blockDim.x;

		if (i > n)
			return;

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
nStreams = 3
n = np.int32(40000)
streamSize = n / nStreams;
streamBytes = streamSize * (32 / 8) # np.float32
bytes = n * 4 # 32/8

gridSize = int(np.ceil(n / blockSize))

# init array A on both host and device
a = []
a_gpu = []

# init STREAMS as array of streams, A and A_GPU arrays where ith element is the partion of the data that streams[i] is assigned to 
streams, events = [], []
for i in range(nStreams):
	streams.append(cuda.Stream())
	
	if (n % nStreams != 0 and i == nStreams-1):
		a.append(np.zeros(n % nStreams + streamSize, dtype=np.float32))
	else:
		a.append(np.zeros(streamSize, dtype=np.float32))

	a_gpu.append(cuda.mem_alloc(a[i].nbytes))
	events.append(dict([("start", cuda.Event()), ("end", cuda.Event())]))

for i in xrange(nStreams + 2):
	ii = i - 1
	iii = i - 2

	if 0 <= iii < nStreams:
		stream = streams[iii]
		cuda.memcpy_dtoh_async(a[iii], a_gpu[iii], stream=stream)
		events[iii]["end"].record(stream)

	if 0 <= ii < nStreams:
		stream = streams[ii]
		kernel(a_gpu[ii], n, grid=(gridSize, 1, 1), block=(blockSize, 1, 1), stream=stream)

	if 0 <= i < nStreams:
		stream = streams[i]
		events[i]["start"].record(stream)
		cuda.memcpy_htod_async(a_gpu[i], a[i], stream=stream)
	
a = np.concatenate(a)

