import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

start = cuda.Event()
end = cuda.Event()

n = np.int32(1e6)
a = np.float32(2)

x = np.ones(n, dtype=np.float32)
y = 2.*np.ones(n, dtype=np.float32)

x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)


# Calculating the time to transfer the data to the device

start.record()

cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)

end.record()
end.synchronize()

print "Transfer to device takes: ", start.time_till(end), " ms"


mod = SourceModule("""
	__global__ void saxpy(int n, float a, float *x, float *y)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n; i += stride)
		{
			y[i] = a*x[i] + y[i];
		}
	}
	""")
dim = (1024, 1, 1)

# Calculating the time to perform the calculation

start.record()

func = mod.get_function("saxpy")
func.prepare(
func(n, a, x_gpu, y_gpu, block=dim)

end.record()
end.synchronize()

print "Calculation on device takes: ", start.time_till(end), " ms"

ans = np.empty_like(y)

# Calculating the time to transfer the data back to the host

start.record()

cuda.memcpy_dtoh(ans, y_gpu)

#print "answer: ", ans

end.record()
end.synchronize()

print "Transfer to host takes: ", start.time_till(end), " ms"

print "Error in calculation is: ", np.max(np.abs(4-ans))
