import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

a = np.random.randn(4,4)
a = a.astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)

print "original array:"
print a

gpu_code = """
    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }
    """


mod = SourceModule(gpu_code)

grid = (1,1)
block = (4, 4, 1)

doublify = mod.get_function("doublify")
doublify.prepare("P")
doublify.prepared_call(grid, block, a_gpu)

print "doubled array:"
print a


