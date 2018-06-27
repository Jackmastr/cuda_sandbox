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

s.record()
[deconv.clean(img, ker) for i in xrange(1000)]
e.record()
e.synchronize()
print "AIPY DECONVOLUTION:", s.time_till(e), "ms"

s.record()
[clean(img, ker) for i in xrange(1000)]
e.record()
e.synchronize()
print "PYCUDA DECONVOLUTION:", s.time_till(e), "ms"