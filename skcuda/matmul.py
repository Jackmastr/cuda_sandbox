import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np 
import skcuda.linalg as linalg
import skcuda.misc as misc
import pycuda.driver as cuda

s = cuda.Event()
e = cuda.Event()
s.record()

linalg.init()
M = 1024
N = 1024

A = np.asarray(np.random.rand(M, N), dtype=np.float32)
B = np.asarray(np.random.rand(N, M), dtype=np.float32)

A_pin = cuda.register_host_memory(A)
B_pin = cuda.register_host_memory(B)

A_gpu = gpuarray.to_gpu(A)
B_gpu = gpuarray.to_gpu(B)
C_gpu = linalg.dot(A_gpu, B_gpu)
#print np.allclose(np.dot(A,B), C_gpu.get())

e.record()
e.synchronize()
print s.time_till(e)/1000., "s"