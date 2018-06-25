import unittest
import numpy as np 
import scipy.signal as sps
import mf

class StreamTest(unittest.TestCase):
	def testStreams(self):
		""" Test using streams """
		in99 = np.random.rand(5,3)
		in0 = np.random.rand(5, 3)
		in1 = np.random.rand(5, 3)
		in2 = np.random.rand(5, 3)
		in3 = np.random.rand(600, 1024)

		check99 = sps.medfilt2d(in99, 1)
		check0 = sps.medfilt2d(in0, 3)
		check1 = sps.medfilt2d(in1, 3)
		check2 = sps.medfilt2d(in2, 3)
		check3 = sps.medfilt2d(in3, 5)

		self.assertTrue(np.allclose(check99, mf.MedianFilter(kernel_size=1, input=[in99]*3)[0]))


		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=3, input=[in0, in1, in0])[0]))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=3, input=[in0, in1, in1])[1]))

		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=3, input=[in99, in0, in1, in2])[2]))
		self.assertTrue(np.allclose(check2, mf.MedianFilter(kernel_size=3, input=[in99, in0, in1, in2])[3]))

		self.assertTrue(np.allclose(check3, mf.MedianFilter(kernel_size=5, input=[in3]*5)[1]))

	def testStreamsMore(self):
		""" A more comprehensive test of streams """
		inList0 = [np.random.rand(100, 67) for i in xrange(10)]

		check0 = [sps.medfilt2d(elem, (5, 7)) for elem in inList0]

		ans0 = mf.MedianFilter(kernel_size=(5,7), input=inList0)

		for i in xrange(10):
			self.assertTrue(np.allclose(check0[i], ans0[i]))



		inList1 = [np.random.rand(600, 1024) for i in xrange(10)]

		check1 = [sps.medfilt2d(elem, (7, 9)) for elem in inList1]

		ans1 = mf.MedianFilter(kernel_size=(7, 9), input=inList1)

		for i in xrange(10):
			self.assertTrue(np.allclose(check1[i], ans1[i]))



if __name__ == '__main__':
	unittest.main()