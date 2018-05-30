import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

first = cuda.Event()
first.record()
final = cuda.Event()

start = cuda.Event()
end = cuda.Event()



n = np.int32(1e8)
a = np.float32(2)

x = np.ones(n, dtype=np.float32)
y = 2.*np.ones(n, dtype=np.float32)

x_pin = cuda.register_host_memory(x)
y_pin = cuda.register_host_memory(y)

x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)

nStreams = 2

stream = [cuda.Stream() for i in range(nStreams)]

# Calculating the time to transfer the data to the device

start.record()

cuda.memcpy_htod_async(x_gpu, x_pin, stream=stream[0])
cuda.memcpy_htod_async(y_gpu, y_pin, stream=stream[1])

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

saxpy = mod.get_function("saxpy")
saxpy.prepare("ifPP")
saxpy.prepared_call((1,1), dim, n, a, x_gpu, y_gpu)

# WITHOUT USING A PREPARED CALL (no noticable speedup here)
#saxpy(n, a, x_gpu, y_gpu, block=dim)



end.record()
end.synchronize()

print "Calculation on device takes: ", start.time_till(end), " ms"

ans = np.empty_like(y)

# Calculating the time to transfer the data back to the host

start.record()

cuda.memcpy_dtoh(y_pin, y_gpu)
ans=y_pin
#print "answer: ", ans

end.record()
end.synchronize()

print "Transfer to host takes: ", start.time_till(end), " ms"

print "Error in calculation is: ", np.max(np.abs(4-ans))

final.record()
print "Total time: ", first.time_till(final), " ms"
