import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import pycuda.driver as cuda 
from skcuda import cublas

s = cuda.Event()
e = cuda.Event()
s.record()

n = np.int32(1e8)
a = np.float32(2)

x = np.ones(n, dtype=np.float32)
y = 2.*np.ones(n, dtype=np.float32)

x_pin = cuda.register_host_memory(x)
y_pin = cuda.register_host_memory(y)

x_gpu = gpuarray.to_gpu(x_pin)
y_gpu = gpuarray.to_gpu(y_pin)

h = cublas.cublasCreate()
cublas.cublasSaxpy(h, x_gpu.size, a, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
cublas.cublasDestroy(h)


# Uncomment to check for errors in the calculation
# np.allclose(y_gpu.get(), a*x + y)

e.record()
e.synchronize()
print s.time_till(e), " ms"