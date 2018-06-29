from CLEAN import clean
import pycuda.driver as cuda
import numpy as np 
from aipy import deconv
import time

s = cuda.Event()
e = cuda.Event()

dim = 1024

img = np.random.rand(dim)
ker = np.random.rand(dim)

# img = np.array([0,0,0,4,6,4,0,0,-2,-3,-2,0]*10, dtype=np.float)
# ker = np.array([3,2,0,0,0,0,0,0,0,0,0,2]*10, dtype=np.float)

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