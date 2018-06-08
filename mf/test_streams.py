import unittest
import numpy as np 
import scipy.signal as sps
import mf_redux

class StreamTest(unittest.TestCase):
	def testStreams(self):
		""" Test using non-square images """
		in0 = np.random.rand(1, 73)
		in1 = np.random.rand(5, 3)
		in2 = np.random.rand(2, 3)
		in3 = np.random.rand(8013, 700)

		check0 = sps.medfilt2d(in0, 1)
		check1 = sps.medfilt2d(in1, 1)
		check2 = sps.medfilt2d(in2, 3)
		check3 = sps.medfilt2d(in3, 5)

		self.assertTrue(np.allclose(check0, mf_redux.MedianFilter(ws=1, n=1, m=73, indata=in0, nStreams=3 )))
		self.assertTrue(np.allclose(check1, mf_redux.MedianFilter(ws=1, n=5, m=3, indata=in1, nStreams=3)))
		self.assertTrue(np.allclose(check2, mf_redux.MedianFilter(ws=3, n=2, m=3, indata=in2, nStreams=6)))
		self.assertTrue(np.allclose(check3, mf_redux.MedianFilter(ws=5, n=8013, m=700, indata=in3, nStreams=10)))

if __name__ == '__main__':
	unittest.main()