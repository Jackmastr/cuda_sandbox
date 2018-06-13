import unittest
import numpy as np 
import scipy.signal as sps
import mf_redux

class DebugTest(unittest.TestCase):
	def testSimple(self):
		""" Test that makes it easier to find the problem in cuda-gdb """
		in0 = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)

		check0 = sps.medfilt2d(in0, (1,3))

		# print check0
		# print mf_redux.MedianFilter(kernel_size=(1, 3), n=4, input=in0)

		self.assertTrue(np.allclose(check0, mf_redux.MedianFilter(kernel_size=(1, 3), n=4, input=in0, bw=1, bh=1)))

if __name__ == '__main__':
	unittest.main()