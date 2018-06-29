from CLEAN import clean
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import unittest
from aipy import deconv

class TestCleanSimple(unittest.TestCase):


	def test_EZ(self):
		img = np.array([0,0,0,4,6,4,0,0,-2,-3,-2,0], dtype=np.float)
		ker = np.array([3,2,0,0,0,0,0,0,0,0,0,2], dtype=np.float)

		# print deconv.clean(img, ker)
		# print clean(img, ker)
		self.assertAlmostEqual(0, clean(img, ker)[0])




if __name__ == '__main__':
	unittest.main()