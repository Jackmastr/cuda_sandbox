import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

import threading

start = cuda.Event()
start.record()
end = cuda.Event()

class GPUThread(threading.Thread):
	def __init__(self, number, n, a, x, y):
		threading.Thread.__init__(self)
		self.number = number
		self.n = n
		self.a = a
		self.x = x
		self.y = y
	def run(self):
		self.dev = cuda.Device(self.number)
		self.ctx = self.dev.make_context()
		
		self.x_gpu = cuda.mem_alloc(x.nbytes)
		self.y_gpu = cuda.mem_alloc(y.nbytes)
		cuda.memcpy_htod(self.x_gpu, x)
		cuda.memcpy_htod(self.y_gpu, y)

		out = test_kernel(self.n, self.a, self.x_gpu, self.y_gpu)
#		print "successful exit from thread ",  self.number
		self.ctx.pop()
		return out
		#del self.x_gpu
		#del self.y_gpu
		#del self.ctx

def test_kernel(n, a, x_gpu, y_gpu):
	code = """
		#include <stdio.h>
		__global__ void saxpy(int n, float a, float *x, float *y)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int i = index; i < n; i += stride)
			{
				y[i] = a*x[i] + y[i];
			}
		}
		"""
	mod = SourceModule(code)
	saxpy = mod.get_function("saxpy")
	saxpy(n, a, x_gpu, y_gpu, block=(1024, 1, 1))
	out = cuda.register_host_memory(np.empty(n, dtype=np.float32))

	cuda.memcpy_dtoh(out, y_gpu)
	return out

n = np.int32(1e8)
a = np.float32(2)
x = np.ones(n, dtype=np.float32)
y = 2.*np.ones(n, dtype=np.float32)

num = cuda.Device.count()
for i in range(num):
	print i
	gpu_thread = GPUThread(i, n, a, x, y)
	out = gpu_thread.run()
	print out
end.record()
end.synchronize()
print "Total time: ", start.time_till(end), " ms"
