import unittest
import numpy as np 
import scipy.signal as sps
import mf_redux

class BigTest(unittest.TestCase):
	# def testFivebyFive(self):
	# 	""" Test using a 5x5 window """
	# 	in0 = np.random.rand(1, 1)
	# 	in1 = np.random.rand(4000, 4000)

	# 	check0 = sps.medfilt2d(in0, (5,5))
	# 	check1 = sps.medfilt2d(in1, (5,5))

	# 	self.assertTrue(np.allclose(check0, mf_redux.MedianFilter(ws=5, n=1, indata=in0)))
	# 	self.assertTrue(np.allclose(check1, mf_redux.MedianFilter(ws=5, n=4000, indata=in1)))
	
	def testNinebyNine(self):
		""" Test using a 9x9 window """
		in0 = np.random.rand(1, 1)
		#in1 = np.random.rand(19, 19)

		check0 = sps.medfilt2d(in0, (9,9))
		#check1 = sps.medfilt2d(in1, (9,9))

		self.assertTrue(np.allclose(check0, mf_redux.MedianFilter(ws=9, n=1, indata=in0, bw=16, bh=16)))
		#self.assertTrue(np.allclose(check1, mf_redux.MedianFilter(ws=9, n=19, indata=in1)))



if __name__ == '__main__':
	unittest.main()