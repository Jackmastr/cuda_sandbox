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

print "one"
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
print "two"
ker = np.array(ker, dtype=np.float32)

area = np.ones(1024)

area0 = area.copy()
area1 = area.copy()
area2 = area.copy()

for i in xrange(len(area)):
	if i in A:
		area1[i] = 0
	else:
		area2[i] = 0

print "three"
class TestOtherSettings(unittest.TestCase):
	def test_tolerances(self):
		print "tol"
		A0 = deconv.clean(img0, ker, stop_if_div=False, tol=0.1, verbose=True)[0]
		# A1 = deconv.clean(img1, ker, stop_if_div=False, tol=1e-6)[0]
		# A2 = deconv.clean(img2, ker, stop_if_div=False, tol=1e-9)[0]

		B0 = clean(img0, ker, stop_if_div=False, tol=0.1)[0]
		# B1 = clean(img1, ker, stop_if_div=False, tol=1e-6)[0]
		# B2 = clean([img2]*3, [ker]*3, stop_if_div=False, tol=1e-9)[0][1]
		# print "four"
		for i in xrange(1024):
			self.assertEqual(A0[i], B0[i])

		# for i in xrange(1024):
		# 	self.assertEqual(A1[i], B1[i])

		# for i in xrange(1024):
		# 	self.assertEqual(A2[i], B2[i])

	# def test_areas(self):
	# 	print "area"
	# 	A0 = deconv.clean(img0, ker, stop_if_div=False, tol=0.1, area=area0)[0]
	# 	A1 = deconv.clean(img1, ker, stop_if_div=False, tol=1e-6, area=area1)[0]
	# 	A2 = deconv.clean(img2, ker, stop_if_div=False, tol=1e-9, area=area2)[0]
	# 	print "area1"
	# 	B0 = clean(img0, ker, stop_if_div=False, tol=0.1, area=area0)[0]
	# 	print "area2"
	# 	B1 = clean(img1, ker, stop_if_div=False, tol=1e-6, area=area1)[0]
	# 	print "area3"
	# 	B2 = clean([img2]*3, [ker]*3, stop_if_div=False, tol=1e-9, area=area2)[0][1]
	# 	print "five"
	# 	for i in xrange(1024):
	# 		self.assertEqual(A0[i], B0[i])

	# 	for i in xrange(1024):
	# 		self.assertEqual(A1[i], B1[i])

	# 	for i in xrange(1024):
	# 		self.assertEqual(A2[i], B2[i])

	def test_mdls(self):
		print "mdl"
		A0 = deconv.clean(img0, ker, stop_if_div=False, tol=0.1, mdl=np.array(area0, dtype=np.float32))[0]
		A1 = deconv.clean(img1, ker, stop_if_div=False, tol=1e-6, mdl=np.array(area1, dtype=np.float32))[0]
		A2 = deconv.clean(img2, ker, stop_if_div=False, tol=1e-9, mdl=np.array(area2, dtype=np.float32))[0]

		B0 = clean(img0, ker, stop_if_div=False, tol=0.1, mdl=area0)[0]
		B1 = clean(img1, ker, stop_if_div=False, tol=1e-6, mdl=area1)[0]
		B2 = clean([img2]*3, [ker]*3, stop_if_div=False, tol=1e-9, mdl=[area2]*3)[0][1]
		print "six"
		for i in xrange(1024):
			self.assertEqual(A0[i], B0[i])

		for i in xrange(1024):
			self.assertEqual(A1[i], B1[i])

		for i in xrange(1024):
			self.assertEqual(A2[i], B2[i])

	# def test_stop_if_div(self):
	# 	A0 = deconv.clean(img0, ker, stop_if_div=True, tol=0.1)[0]
	# 	A1 = deconv.clean(img1, ker, stop_if_div=True, tol=1e-6)[0]
	# 	A2 = deconv.clean(img2, ker, stop_if_div=True, tol=1e-9)[0]

	# 	B0 = clean(img0, ker, stop_if_div=True, tol=0.1)[0]
	# 	B1 = clean(img1, ker, stop_if_div=True, tol=1e-6)[0]
	# 	B2 = clean([img2]*3, [ker]*3, stop_if_div=True, tol=1e-9)[0][1]

	# def test_gain(self):
	# 	A0 = deconv.clean(img0, ker, stop_if_div=False, tol=0.1, gain=0.01)[0]
	# 	A1 = deconv.clean(img1, ker, stop_if_div=False, tol=1e-6, gain=0.001)[0]
	# 	A2 = deconv.clean(img2, ker, stop_if_div=False, tol=1e-9, gain=0.3)[0]

	# 	B0 = clean(img0, ker, stop_if_div=False, tol=0.1, gain=0.01)[0]
	# 	B1 = clean(img1, ker, stop_if_div=False, tol=1e-6, gain=0.001)[0]
	# 	B2 = clean([img2]*3, [ker]*3, stop_if_div=False, tol=1e-9, gain=0.3)[0][1]



if __name__ == '__main__':
	unittest.main()