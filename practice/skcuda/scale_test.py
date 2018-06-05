import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np 
import skcuda.linalg as linalg
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

s = cuda.Event()
e = cuda.Event()

N = np.int32(1<<20)
alpha = np.float32(2.5)

x = np.asarray(np.random.rand(N), np.float32)

s.record()
linalg.init()

x_gpu = gpuarray.to_gpu(x)
linalg.scale(alpha, x_gpu)
ans = x_gpu.get()
e.record()
e.synchronize()
print s.time_till(e), "ms for skcuda"


s.record()

code = """
	__global__ void scale(int N, float alpha, float* x)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		for (int i = index; i < N; i += stride)
		{
			x[i] = x[i] * alpha;
		}
	}
	"""
mod = SourceModule(code)
scale = mod.get_function("scale")

x_pin = cuda.register_host_memory(x)
x_gpu = cuda.mem_alloc(x.nbytes)
cuda.memcpy_htod(x_gpu, x)

scale.prepare("ifP")
scale.prepared_call((1, 1), (1024, 1, 1), N, alpha, x_gpu)

b = np.zeros_like(x)
cuda.memcpy_dtoh(b, x_gpu)

e.record()
e.synchronize()
print s.time_till(e), "ms for vanilla pycuda"
print np.max(np.abs(b - alpha*x))
# print b
# print x