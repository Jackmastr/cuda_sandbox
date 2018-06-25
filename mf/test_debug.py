import unittest
import numpy as np 
import scipy.signal as sps
import mf

class DebugTest(unittest.TestCase):
	def testSimple(self):
		""" Test that makes it easier to find the problem in cuda-gdb """
		in0 = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)

		check0 = sps.medfilt2d(in0, 3)

		print check0
		print mf.MedianFilter(kernel_size=3, input=[in0]*3)[0]

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=3, input=[in0, in0, in0])[0]))

if __name__ == '__main__':
	unittest.main()