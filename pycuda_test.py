# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy
a = numpy.random.randn(4,4)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)

xdim = 2

mod = SourceModule("""
	__global__ void doublify(float *a)
	{
		int idx = threadIdx.x + threadIdx.y*%d;
		a[idx] *= 2;
	}
	""" % xdim)

func = mod.get_function("doublify")
func(a_gpu, block=(xdim,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print "original array:"
print a
print "doubled with kernel:"
print a_doubled
