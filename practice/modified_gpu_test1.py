import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 

# Aaron's base code ~630ms
# My code when I substitute out the gpuarrays --> ~390ms

s = cuda.Event()
e = cuda.Event()
s.record()

code = """
	__global__ void custom_kernel(float *g_y, float *g_x)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		const float x = g_x[i];
		g_y[i] = cos(x)*exp(sin(x)-sqrt(x*x));
	}
	"""
mod = SourceModule(code)
custom_kernel = mod.get_function("custom_kernel");
size = 5120000
block_size = 512
grid_size = size/block_size
block = (block_size, 1, 1)
grid = (grid_size, 1)

x = np.linspace(1, size, size).astype(np.float32)
print x.shape

x_pin = cuda.register_host_memory(x)

x_gpu = cuda.mem_alloc(x.nbytes)
cuda.memcpy_htod(x_gpu, x)


y_gpu = cuda.mem_alloc(x.nbytes)


custom_kernel(y_gpu, x_gpu, block=block, grid=grid)

ans = np.zeros_like(x_pin)
cuda.memcpy_dtoh(ans, y_gpu)
ans = np.sum(ans)

print ans
print np.sum(np.cos(x) * np.exp(np.sin(x) - np.sqrt(x*x)))

e.record()
e.synchronize()
print s.time_till(e), "ms"