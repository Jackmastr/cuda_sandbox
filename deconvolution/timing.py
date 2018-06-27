from CLEAN import clean
import pycuda.driver as cuda
import numpy as np 
from aipy import deconv

s = cuda.Event()
e = cuda.Event()

dim = 2**20

img = np.random.rand(dim)
ker = np.random.rand(dim)

s.record()
deconv.clean(img, ker)
e.record()
e.synchronize()
print "AIPY DECONVOLUTION:", s.time_till(e), "ms"

s.record()
clean(img, ker)
e.record()
e.synchronize()
print "PYCUDA DECONVOLUTION:", s.time_till(e), "ms"