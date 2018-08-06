from CLEAN import clean
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import unittest
from aipy import deconv
import warnings
warnings.filterwarnings("ignore")
class TestCleanSimple(unittest.TestCase):
	def test_ZeroGain(self):
		dim = 128
		res = np.ones(dim)
		ker = np.ones(dim)
		mdl = np.ones(dim)
		area = np.ones(dim)
		self.assertEqual(1, clean(res, ker, mdl, area, 0, 100, 1, 0, 1024)[0][0])
		self.assertEqual(1, clean(res, ker, mdl, area, 0, 100, 1, 0, 1024)[0][-1])

	def test_Default(self):
		img = np.array([0,0,0,4,6,4,0,0,-2,-3,-2,0], dtype=np.float32)
		ker = np.array([3,2,0,0,0,0,0,0,0,0,0,2], dtype=np.float32)

		#print deconv.clean(img, ker, verbose=True)[0]
		#print clean(img, ker)[0]


		for i in xrange(12):
			self.assertEqual(deconv.clean(img, ker)[0][i], clean(img, ker)[0][i])

	def test_RandomInput(self):
		dim = 25
		img = np.array(np.random.rand(dim), dtype=np.float32)
		ker = np.array(np.random.rand(dim), dtype=np.float32)

		A = deconv.clean(img, ker)[0]
		B = clean(img, ker)[0]

		for i in xrange(dim):
			self.assertEqual(A[i], B[i])



if __name__ == '__main__':
	unittest.main()