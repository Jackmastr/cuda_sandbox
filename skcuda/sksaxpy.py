import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import pycuda.driver as cuda 
from skcuda import cublas

s = cuda.Event()
e = cuda.Event()
s.record()

N = np.int32(1e8)
a = np.float32(2)

x = np.ones(N, dtype=np.float32)
y = 2.*np.ones(N, dtype=np.float32)

nStreams = 2
streams = [cuda.Stream() for i in range(nStreams)]

x_pin = [cuda.register_host_memory(x[i*N/nStreams:(i+1)*N/nStreams]) for i in range(nStreams)]
y_pin = [cuda.register_host_memory(y[i*N/nStreams:(i+1)*N/nStreams]) for i in range(nStreams)]

h = cublas.cublasCreate()

x_gpu = np.empty(nStreams, dtype=object)
y_gpu = np.empty(nStreams, dtype=object)
ans = np.empty(nStreams, dtype=object)

for i in range(nStreams):
	cublas.cublasSetStream(h, streams[i].handle)
	
	x_gpu[i] = gpuarray.to_gpu_async(x_pin[i], stream=streams[i])
	y_gpu[i] = gpuarray.to_gpu_async(y_pin[i], stream=streams[i]) 

	cublas.cublasSaxpy(h, x_gpu[i].size, a, x_gpu[i].gpudata, 1, y_gpu[i].gpudata, 1)
	ans[i] = y_gpu[i].get_async(stream=streams[i])


cublas.cublasDestroy(h)


# Uncomment to check for errors in the calculation
#y_gpu = np.array([yg.get() for yg in y_gpu])
#y_gpu = np.array(y_gpu).reshape(y.shape)
#print np.allclose(y_gpu, a*x + y)

e.record()
e.synchronize()
print s.time_till(e), " ms"