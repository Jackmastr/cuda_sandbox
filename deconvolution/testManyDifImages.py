from CLEAN import clean
import pycuda.driver as cuda
import numpy as np 
from aipy import deconv
import time
import random

import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import unittest
from hera_sim import noise, foregrounds
import aipy

import warnings
warnings.filterwarnings("ignore")

fqs = np.linspace(.1,.2,1024,endpoint=False)
lsts = np.linspace(0,2*np.pi,10000, endpoint=False)
bl_len_ns = 30.
vis_fg_pntsrc = foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=200)
img0 = np.array(vis_fg_pntsrc[100], dtype=np.float32)
img1 = np.array(vis_fg_pntsrc[200], dtype=np.float32)
img2 = np.array(vis_fg_pntsrc[300], dtype=np.float32)
img3 = np.array(vis_fg_pntsrc[400], dtype=np.float32)
img4 = np.array(vis_fg_pntsrc[500], dtype=np.float32)
img5 = np.array(vis_fg_pntsrc[600], dtype=np.float32)
img6 = np.array(vis_fg_pntsrc[700], dtype=np.float32)
img7 = np.array(vis_fg_pntsrc[800], dtype=np.float32)
img8 = np.array(vis_fg_pntsrc[900], dtype=np.float32)

ker0 = np.ones(1024)
ker1 = np.ones(1024)

A = set()
while len(A) < 160:
	A.add(random.randint(0, 1024))
for i in xrange(len(ker0)):
	if i in A:
		ker0[i] = 0

A = set()
while len(A) < 160:
	A.add(random.randint(0, 1024))
for i in xrange(len(ker1)):
	if i in A:
		ker1[i] = 0

ker0 = np.fft.fft(ker0)
ker1 = np.fft.fft(ker1)

ker0 = np.array(ker0, dtype=np.float32)
ker1 = np.array(ker1, dtype=np.float32)

kers = [ker0, ker1, ker0, ker1, ker0, ker1, ker0, ker1, ker0]
imgs = [img0, img1, img2, img3, img4, img5, img6, img7, img8]

class TestManyDifImages(unittest.TestCase):
	def test_many(self):
		A0 = deconv.clean(img0, ker0, stop_if_div=False, tol=0)[0]
		A1 = deconv.clean(img1, ker1, stop_if_div=False, tol=0)[0]
		A2 = deconv.clean(img2, ker0, stop_if_div=False, tol=0)[0]
		A3 = deconv.clean(img3, ker1, stop_if_div=False, tol=0)[0]
		A4 = deconv.clean(img4, ker0, stop_if_div=False, tol=0)[0]
		A5 = deconv.clean(img5, ker1, stop_if_div=False, tol=0)[0]
		A6 = deconv.clean(img6, ker0, stop_if_div=False, tol=0)[0]
		A7 = deconv.clean(img7, ker1, stop_if_div=False, tol=0)[0]
		A8 = deconv.clean(img8, ker0, stop_if_div=False, tol=0)[0]

		B = clean(imgs, kers, stop_if_div=False, tol=0)[0]

		for i in xrange(1024):
			self.assertAlmostEqual(A5[i], B[5][i])




if __name__ == '__main__':
	unittest.main()