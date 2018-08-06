from CLEAN import clean
import pycuda.driver as cuda
import numpy as np 
from aipy import deconv
import time
from hera_sim import foregrounds
import random

import warnings
warnings.filterwarnings("ignore")

data_type = np.complex64

fqs = np.linspace(.1,.2,1024,endpoint=False)
lsts = np.linspace(0,2*np.pi,10000, endpoint=False)
bl_len_ns = 30.
vis_fg_pntsrc = foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=200)
img0 = np.array(vis_fg_pntsrc[100], dtype=data_type)
img1 = np.array(vis_fg_pntsrc[500], dtype=data_type)
img2 = np.array(vis_fg_pntsrc[700], dtype=data_type)
ker = np.ones(1024)

A = set()
while len(A) < 160:
	A.add(random.randint(0, 1024))
for i in xrange(len(ker)):
	if i in A:
		ker[i] = 0

ker = np.fft.fft(ker)

ker = np.array(ker, dtype=data_type)


numImgs = 61250
# numImgs = 4096

# print "AIPY CLEAN"

# s = cuda.Event()
# e = cuda.Event()
# s.record()
# for i in xrange(numImgs):
# 	deconv.clean(img1, ker, stop_if_div=False, tol=0)
# e.record()
# e.synchronize()
# print s.time_till(e)/1000, "s"

# s = cuda.Event()
# e = cuda.Event()
# s.record()
# deconv.clean(img1, ker, stop_if_div=False, tol=0)
# e.record()
# e.synchronize()
# print s.time_till(e), "ms"

# s = cuda.Event()
# e = cuda.Event()
# s.record()
# deconv.clean(img2, ker, stop_if_div=False, tol=0, maxiter=100000)
# e.record()
# e.synchronize()
# print s.time_till(e), "ms"

print "MY CLEAN"

s = cuda.Event()
e = cuda.Event()
s.record()
clean([img1]*numImgs, [ker]*numImgs, stop_if_div=False, tol=0)
e.record()
e.synchronize()
print s.time_till(e)/1000, "s"

# s = cuda.Event()
# e = cuda.Event()
# s.record()
# clean([img1]*numImgs, [ker]*numImgs, stop_if_div=False, tol=0, maxiter=1000)
# e.record()
# e.synchronize()
# print s.time_till(e)/1000, "s"


# ---REAL---
# 10 images

# AIPY CLEAN
# 5.60483349609 s
# MY CLEAN
# 89.616 s

# AIPY CLEAN
# 7.00130859375 s
# MY CLEAN
# 107.249953125 s

# 100 Images

# AIPY CLEAN
# 69.4694765625 s
# MY CLEAN
# 217.042171875 s

# AIPY CLEAN
# 67.9782890625 s
# MY CLEAN
# 211.066609375 s



# 1000 Images

# AIPY CLEAN
# 515.94825 s
# MY CLEAN
# 511.08453125 s


# ---COMPLEX---
# 10 Images

# AIPY CLEAN
# 7.40669091797 s
# MY CLEAN
# 72.8940078125 s

# AIPY CLEAN
# 5.69029345703 s
# MY CLEAN
# 55.2136445312 s

# 100 Images

# AIPY CLEAN
# 59.6129414063 s
# MY CLEAN
# 124.54153125 s

# AIPY CLEAN
# 50.2183867187 s
# MY CLEAN
# 107.356015625 s


# 1000 Images

# AIPY CLEAN
# 553.178875 s
# MY CLEAN
# 479.81659375 s

# 5000 Images

# MY CLEAN
# 592.8794375 s

# 10,000 Images

# MY CLEAN
# 926.5141875 s

# 350^2/2 Images

