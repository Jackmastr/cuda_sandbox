from CLEAN import clean
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import unittest

DIM = 128

class TestCleanSimple(unittest.TestCase):
	def test_ZeroGain(self):
		res = np.ones(DIM)
		ker = np.ones(DIM)
		mdl = np.ones(DIM)
		area = np.ones(DIM)
		self.assertEqual(1, clean(res, ker, mdl, area, 0, 100, 1, 0, 1024)[0])
		self.assertEqual(1, clean(res, ker, mdl, area, 0, 100, 1, 0, 1024)[-1])

if __name__ == '__main__':
	unittest.main()