#!/usr/bin/env python

# maybe try pragma loop unrolls???

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import skcuda

def clean(res, ker, mdl=None, area=None, gain=0.1, maxiter=10000, tol=1e-3, stop_if_div=1, blockDimX=512):
	s = cuda.Event()
	e = cuda.Event()

	gain = np.float64(gain)
	maxiter = np.int32(maxiter)
	tol = np.float64(tol)
	stop_if_div = np.int32(stop_if_div)

	res = np.array(res, dtype=np.float32)
	ker = np.array(ker, dtype=np.float32)

	dim = np.int32(len(res))

	if mdl is None:
		mdl = np.array(np.zeros(dim), dtype=np.float32)
	else:
		mdl = np.array(mdl, dtype=np.float32)
	
	if area is None:
		area = np.array(np.ones(dim), dtype=np.float32)
	else:
		area = np.array(area, dtype=np.int32)

	block = (blockDimX, 1, 1)
	grid = (int(dim/blockDimX)+1, 1, 1)

	#block=(1,1,1)
	#grid=(1,1,1)

	code = """
		__global__ void compute_res(float *res, float *ker, float step, float *square_res)
		{
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index < %(DIM)s)
			{
				int wrapped_index = index %% %(DIM)s;
				res[wrapped_index] -= ker[index] * step;
				res_at_wrapped = res[wrapped_index];
				square_res_gpu[wrapped_index] = res_at_wrapped * res_at_wrapped;
			}
		}

	
	"""
	code = code % {
		'DIM': dim,
		'GAIN': gain,
	}

	mod = SourceModule(code)
	compute_res = mod.get_function("compute_res")


	res_pin = cuda.register_host_memory(res)
	ker_pin = cuda.register_host_memory(ker)
	mdl_pin = cuda.register_host_memory(mdl)
	area_pin = cuda.register_host_memory(area)

	res_gpu = cuda.mem_alloc(res.nbytes)
	ker_gpu = cuda.mem_alloc(ker.nbytes)
	mdl_gpu = cuda.mem_alloc(mdl.nbytes)
	area_gpu = cuda.mem_alloc(area.nbytes)

	# Additional buffers that may be needed
	square_res_gpu = cuda.mem_alloc(res.nbytes)

	cuda.memcpy_htod(res_gpu, res_pin)
	cuda.memcpy_htod(ker_gpu, ker_pin)
	cuda.memcpy_htod(mdl_gpu, mdl_pin)
	cuda.memcpy_htod(area_gpu, area_pin)

	# Compute the gain/phase of the kernel to start out

	for n in xrange(maxiter):

		compute_res(res_gpu, ker_gpu, step, square_res_gpu)

		square_res_gpu = skcuda.misc.cumsum(square_res_gpu)



