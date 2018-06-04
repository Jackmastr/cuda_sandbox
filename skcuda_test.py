import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np 
import skcuda.linalg as linalg

s = cuda.Event()
e = cuda.Event()

s.record()

N = 32 * 1024

linalg.init()
a = np.tril(np.ones(N, dtype=np.float32))
a_gpu = gpuarray.to_gpu(a)
at_gpu = linalg.transpose(a_gpu)
print "done"
e.record()
e.synchronize()
print s.time_till(e)