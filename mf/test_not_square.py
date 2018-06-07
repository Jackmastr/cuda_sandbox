import unittest
import numpy as np 
import scipy.signal as sps
import mf_redux

class BasicTest(unittest.TestCase):
	def testLopsidedImage(self):
		""" Test using non-square images """
		in0 = np.random.rand(1, 73)
		in1 = np.random.rand(29,12)
		in2 = np.random.rand(44, 45)
		in3 = np.random.rand(700, 8013)

		check0 = sps.medfilt2d(in0, (1,1))
		check1 = sps.medfilt2d(in1, (3,3))
		check2 = sps.medfilt2d(in2, (3,3))
		check3 = sps.medfilt2d(in3, (3,3))

		self.assertTrue(np.allclose(check0, mf_redux.MedianFilter(ws=1, n=1, m=73, indata=in0)))


		#print check1
		#print mf_redux.MedianFilter(ws=3, n=29, m=12, indata=in1)


		self.assertTrue(np.allclose(check1, mf_redux.MedianFilter(ws=3, n=29, m=12, indata=in1)))
		self.assertTrue(np.allclose(check2, mf_redux.MedianFilter(ws=3, n=44, m=45, indata=in2)))
		self.assertTrue(np.allclose(check3, mf_redux.MedianFilter(ws=3, n=700, m=8013, indata=in3)))

if __name__ == '__main__':
	unittest.main()