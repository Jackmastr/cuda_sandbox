import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand

add = ElementwiseKernel("float *a, float *b, float *c", "c[i] = a[i] + b[i]", "add")

shape = 128, 1024
a_gpu = curand(shape)
b_gpu = curand(shape)

c_gpu = gpuarray.empty_like(a_gpu)
add(a_gpu, b_gpu, c_gpu)

print np.max(np.abs(c_gpu.get() - a_gpu.get() - b_gpu.get()))
