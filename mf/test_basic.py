import unittest
import numpy as np 
import scipy.signal as sps
import mf

class BasicTest(unittest.TestCase):
	def testOneByOne(self):
		""" Test using a 1x1 window """
		in0 = np.random.rand(1, 1)
		in1 = np.random.rand(29, 29)
		in2 = np.random.rand(93, 93)
		in3 = np.random.rand(7000, 7000)

		check0 = sps.medfilt2d(in0, 1)
		check1 = sps.medfilt2d(in1, 1)
		check2 = sps.medfilt2d(in2, 1)
		check3 = sps.medfilt2d(in3, 1)

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=1, input=in0)))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=1, input=in1)))
		self.assertTrue(np.allclose(check2, mf.MedianFilter(kernel_size=1, input=in2)))
		self.assertTrue(np.allclose(check3, mf.MedianFilter(kernel_size=1, input=in3)))

	def testThreeByThree(self):
		""" Test using a 3x3 window """
		in0 = np.random.rand(1, 1)
		in1 = np.random.rand(29, 29)
		in2 = np.random.rand(93, 93)
		in3 = np.random.rand(7000, 7000)

		check0 = sps.medfilt2d(in0, 3)
		check1 = sps.medfilt2d(in1, 3)
		check2 = sps.medfilt2d(in2, 3)
		check3 = sps.medfilt2d(in3, 3)

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=3, input=in0)))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=3, input=in1)))
		self.assertTrue(np.allclose(check2, mf.MedianFilter(kernel_size=3, input=in2)))
		self.assertTrue(np.allclose(check3, mf.MedianFilter(kernel_size=3, input=in3)))

if __name__ == '__main__':
	unittest.main()