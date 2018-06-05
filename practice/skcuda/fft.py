import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np 
from skcuda.fft import fft, Plan
import pycuda.driver as cuda
import skcuda.cublas as cublas
import skcuda

s = cuda.Event()
e = cuda.Event()
s.record()

nStreams = 8
stream = [cuda.Stream() for i in range(nStreams)]
N = 8192
print skcuda.misc.get_current_device()

x = [np.asarray(np.random.rand(N/nStreams), np.float32) for i in range(nStreams)]
#x_pin = cuda.register_host_memory(x)
#xf = np.fft.fft(x)
x_gpu = [gpuarray.to_gpu_async(x[i], stream=stream[i]) for i in range(nStreams)]

xf_gpu = [gpuarray.empty((N/nStreams)/2 + 1, np.complex64) for i in range(nStreams)]
plan = [Plan(x[0].shape, np.float32, np.complex64, stream=stream[i]) for i in range(nStreams)]
print skcuda.misc.get_current_device()
for i in range(nStreams):
	fft(x_gpu[i], xf_gpu[i], plan[i])
	print skcuda.misc.get_current_device()

x_pin = [xf_gpu[i].get_async(stream=stream[i]) for i in range(nStreams)]

#print np.allclose(xf[0:N/2 + 1], xf_gpu.get(), atol=1e-6)

e.record()
e.synchronize()
print s.time_till(e), "ms"