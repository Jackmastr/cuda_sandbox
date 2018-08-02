from CLEAN import clean
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import unittest
from hera_sim import noise, foregrounds
import aipy
from aipy import deconv
import random

import warnings
warnings.filterwarnings("ignore")

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
		
		ker = np.array(np.fft.fft(ker), dtype=np.float32)

		A0, info = deconv.clean(img0, ker, stop_if_div=True, tol=0, verbose=True)
		# A1 = deconv.clean(img1, ker, stop_if_div=True, tol=0)[0]
		# A2 = deconv.clean(img2, ker, stop_if_div=True, tol=0)[0]

		B0 = clean(img0, ker, stop_if_div=True, tol=0)[0]
		# B1 = clean(img1, ker, stop_if_div=True, tol=0)[0]
		# B2 = clean([img2]*3, [ker]*3, stop_if_div=False, tol=0)[0][1]

		for i in xrange(1024):
			self.assertEqual(A0[i], B0[i])

		# for i in xrange(1024):
		# 	self.assertEqual(A1[i], B1[i])

		# for i in xrange(1024):
		# 	self.assertEqual(A2[i], B2[i])


	def test_spike(self):
		ker = np.zeros(1024, dtype=np.float32)
		ker[0] = 1
		img = ker.copy()
		A, info = deconv.clean(img, ker, stop_if_div=True, tol=0, maxiter=int(1e4))

		B = clean(img, ker, stop_if_div=True, tol=0, maxiter=int(1e4))[0]

		for i in xrange(1024):
			self.assertEqual(A[i], B[i])


	def test_spikes(self):
		ker = np.zeros(1024, dtype=np.float32)
		ker[0] = 1
		img = ker.copy()
		A, info = deconv.clean(img, ker, stop_if_div=True, tol=0, maxiter=int(1e4), verbose=True)

		B = clean([img]*10, [ker]*10, stop_if_div=True, tol=0, maxiter=int(1e4))[0]

		for i in xrange(1024):
			self.assertEqual(A[i], B[6][i])





if __name__ == '__main__':
	unittest.main()