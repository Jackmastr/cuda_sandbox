#from CLEAN import clean
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import unittest
import aipy
from aipy import deconv
import hera_sim
from hera_sim import noise, foregrounds

class TestHera(unittest.TestCase):
	def test_foregrounds(self):
		fqs = np.linspace(.1,.2,1024,endpoint=False)
		lsts = np.linspace(0,2*np.pi,10000, endpoint=False)
		times = lsts / (2*np.pi) * aipy.const.sidereal_day
		bl_len_ns = 30.

		Tsky_mdl = noise.HERA_Tsky_mdl['xx']
		vis_fg_diffuse = foregrounds.diffuse_foreground(Tsky_mdl, lsts, fqs, bl_len_ns)
		print vis_fg_diffuse


if __name__ == '__main__':
	unittest.main()