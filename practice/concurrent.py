import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
t = time.time()

code = """
	#include <stdio.h>
	__global__ void my_kernel(float *d)
	{
		const int i = threadIdx.x;
		for (int m=0; m < (100000000); m++)
		{
			d[i] *= 2.0;
			d[i] /= 2.0;
		}
		d[i] = d[i] * 2.0;
	}
	"""
mod = SourceModule(code)
my_kernel = mod.get_function("my_kernel")

# Create the test data on host
N = 400 # size of each dataset
n = 2 # number of datasets
data, data_check, d_data = [], [], []
for k in range(n):
	data.append(np.random.randn(N).astype(np.float32))
	data_check.append(data[k].copy())
	d_data.append(cuda.mem_alloc(data[k].nbytes))

ref = cuda.Event()
ref.record()

stream, event = [], []
marker_names = ['kernel_begin', 'kernel_end']
for k in range(n):
	stream.append(cuda.Stream())
	event.append(dict([(marker_names[i], cuda.Event()) for i in range(len(marker_names))]))

for k in range(n):
	cuda.memcpy_htod(d_data[k], data[k])

for k in range(n):
	event[k]['kernel_begin'].record(stream[k])
	my_kernel(d_data[k], block=(N,1,1), stream=stream[k])
	for k in range(n): #commenting out breaks concurrency. BUT WHY?
		event[k]['kernel_end'].record(stream[k])

for k in range(n):
	cuda.memcpy_dtoh(data[k], d_data[k])

for k in range(n):
	print np.max(data[k] - (data_check[k]*2))

print (time.time() - t)
