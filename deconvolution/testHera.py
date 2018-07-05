#from CLEAN import clean
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import unittest
from hera_sim import noise, foregrounds
import aipy
from aipy import deconv
import random

class TestHera(unittest.TestCase):
	def test_foregrounds(self):
		fqs = np.linspace(.1,.2,1024,endpoint=False)
		lsts = np.linspace(0,2*np.pi,10000, endpoint=False)
		bl_len_ns = 30.
		vis_fg_pntsrc = foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=200)
		img0 = np.array(vis_fg_pntsrc[100], dtype=np.float32)
		img1 = np.array(vis_fg_pntsrc[500], dtype=np.float32)
		img2 = np.array(vis_fg_pntsrc[700], dtype=np.float32)
		ker = np.ones(1024)
		
		A = set()
		while len(A) < 160:
			A.add(random.randint(0, 1024))
		for i in xrange(len(ker)):
			if i in A:
				ker[i] = 0
		
		ker = np.array(ker, dtype=np.float32)

		s = cuda.Event()
		e = cuda.Event()
		s.record()
		deconv.clean(img0, ker, stop_if_div=False)
		e.record()
		e.synchronize()
		print s.time_till(e), "ms"

		s = cuda.Event()
		e = cuda.Event()
		s.record()
		deconv.clean(img1, ker, stop_if_div=False)
		e.record()
		e.synchronize()
		print s.time_till(e), "ms"

		s = cuda.Event()
		e = cuda.Event()
		s.record()
		deconv.clean(img2, ker, stop_if_div=False)
		e.record()
		e.synchronize()
		print s.time_till(e), "ms"




if __name__ == '__main__':
	unittest.main()