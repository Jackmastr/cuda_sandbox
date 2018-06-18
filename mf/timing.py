#!/usr/bin/env python

import numpy as np 
import scipy.signal as sps
import mf_redux
import pycuda.driver as cuda

s = cuda.Event()
e = cuda.Event()
inList = [np.random.rand(600, 1024) for i in xrange(10)]
inListBig = [np.random.rand(600, 1024) for i in xrange(100)]
#inListHuge = [np.random.rand(600, 1024) for i in xrange(5000)]
print "done making it"

s.record()
[sps.medfilt2d(elem, (17, 17)) for elem in inList]
e.record()
e.synchronize()
print "SCIPY MEDFILT w/ 10 IMAGES: ", s.time_till(e), "ms"

s.record()
mf_redux.MedianFilter(kernel_size=(17, 17), input_list=inList)
e.record()
e.synchronize()
print "THIS MEDFILT w/ 10 IMAGES: ", s.time_till(e), "ms"

# s.record()
# [sps.medfilt2d(elem, (17, 17)) for elem in inListBig]
# e.record()
# e.synchronize()
# print "SCIPY MEDFILT w/ 100 IMAGES: ", s.time_till(e), "ms"

s.record()
mf_redux.MedianFilter(kernel_size=(17, 17), input_list=inListBig)
e.record()
e.synchronize()
print "THIS MEDFILT w/ 100 IMAGES: ", s.time_till(e), "ms"

#s.record()
#mf_redux.MedianFilter(kernel_size=(17, 17), input_list=inListHuge)
#e.record()
#e.synchronize()
#print "THIS MEDFILT w/ 5000 IMAGES: ", s.time_till(e), "ms"


# input0 = np.random.rand(600, 1024)
# mf_redux.MedianFilter(kernel_size=11, input=input0)