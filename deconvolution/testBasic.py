from CLEAN import clean
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import unittest
from aipy import deconv

class TestCleanSimple(unittest.TestCase):
	def test_ZeroGain(self):
		dim = 128
		res = np.ones(dim)
		ker = np.ones(dim)
		mdl = np.ones(dim)
		area = np.ones(dim)
		self.assertEqual(1, clean(res, ker, mdl, area, 0, 100, 1, 0, 1024)[0])
		self.assertEqual(1, clean(res, ker, mdl, area, 0, 100, 1, 0, 1024)[-1])

	def test_Default(self):
		img = np.array([0,0,0,4,6,4,0,0,-2,-3,-2,0], dtype=np.float)
		ker = np.array([3,2,0,0,0,0,0,0,0,0,0,2], dtype=np.float)

		for i in xrange(12):
			self.assertAlmostEqual(deconv.clean(img, ker)[0][i], clean(img, ker)[i])

	def test_RandomInput(self):
		dim = 25
		img = np.random.rand(dim)
		ker = np.random.rand(dim)

		for i in xrange(dim):
			self.assertAlmostEqual(deconv.clean(img, ker)[0][i], clean(img, ker)[i])



if __name__ == '__main__':
	unittest.main()