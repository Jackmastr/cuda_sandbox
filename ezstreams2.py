import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.gpuarray as gpuarray

s = cuda.Event()
e = cuda.Event()
s.record()

code = """
	__global__ void add_one(int n, int start, float *x)
	{
		int index = start + blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		for (int i = index; i < n; i += stride)
			x[i] += 1.0;
	}
	"""
mod = SourceModule(code)
add_one = mod.get_function("add_one")

N = np.int32(1e8)
nStreams = 2
streamSize = np.int32(N/nStreams)

x = [None]*nStreams
x_gpu = [None]*nStreams
for i in range(nStreams):
	x[i] = np.ones(streamSize, dtype=np.float32)
	x_gpu[i] = cuda.mem_alloc(x[i].nbytes)
	cuda.memcpy_htod(x_gpu[i], x[i])

#cuda.memcpy_htod(x_gpu, x)

stream = []
for i in range(nStreams):
	stream.append(cuda.Stream())


for i in range(nStreams):
	start = np.int32(i*streamSize)
	stop = np.int32((i+1)*streamSize)
	add_one(stop, start, x_gpu[i], block=(1024, 1, 1), stream=stream[i])


ans = [None]*nStreams
for i in range(nStreams):
	ans[i] = np.empty_like(x[i])
	print i
	cuda.memcpy_dtoh(ans[i], x_gpu[i])
ans = (np.asarray(ans)).flatten()

print np.where(ans == 1)

e.record()
print s.time_till(e)
