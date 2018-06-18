import unittest
import numpy as np 
import scipy.signal as sps
import mf_redux

class NotSquareTest(unittest.TestCase):
	def testLopsidedImage(self):
		""" Test using non-square images """
		in0 = np.random.rand(1, 73)
		in1 = np.random.rand(5, 3)
		in2 = np.random.rand(2, 3)
		in3 = np.random.rand(8013, 700)

		check0 = sps.medfilt2d(in0, 1)
		check1 = sps.medfilt2d(in1, 1)
		check2 = sps.medfilt2d(in2, 3)
		check3 = sps.medfilt2d(in3, 5)

		self.assertTrue(np.allclose(check0, mf_redux.MedianFilter(kernel_size=1, input_list=[in0])))
		self.assertTrue(np.allclose(check1, mf_redux.MedianFilter(kernel_size=1, input_list=[in1])))
		self.assertTrue(np.allclose(check2, mf_redux.MedianFilter(kernel_size=3, input_list=[in2])))
		self.assertTrue(np.allclose(check3, mf_redux.MedianFilter(kernel_size=5, input_list=[in3])))

	def testLopsidedWindow(self):
		""" Test using non-square windows """
		in0 = np.random.rand(1, 73)
		in1 = np.random.rand(5, 3)
		in2 = np.random.rand(2, 3)
		in3 = np.random.rand(8013, 700)

		check0 = sps.medfilt2d(in0, (1, 11))
		check1 = sps.medfilt2d(in1, (11, 1))
		check2 = sps.medfilt2d(in2, (3, 5))
		check3 = sps.medfilt2d(in3, (9, 5))

		self.assertTrue(np.allclose(check0, mf_redux.MedianFilter(kernel_size=(1, 11), input_list=[in0])))
		self.assertTrue(np.allclose(check1, mf_redux.MedianFilter(kernel_size=(11, 1), input_list=[in1])))
		self.assertTrue(np.allclose(check2, mf_redux.MedianFilter(kernel_size=(3, 5), input_list=[in2])))
		self.assertTrue(np.allclose(check3, mf_redux.MedianFilter(kernel_size=(9, 5), input_list=[in3])))


if __name__ == '__main__':
	unittest.main()