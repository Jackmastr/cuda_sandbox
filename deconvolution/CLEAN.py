#!/usr/bin/env python

# maybe try pragma loop unrolls???

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import skcuda
from skcuda import cublas
import pycuda.gpuarray as gpuarray
from math import sqrt

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
		__global__ void compute_res(float *res, float *ker, float step, float *square_res, int* area)
		{
			const int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index < %(DIM)s && area[index])
			{
				int wrapped_index = index %% %(DIM)s;
				res[wrapped_index] -= ker[index] * step;
				res_at_wrapped = res[wrapped_index];
				square_res[wrapped_index] = res_at_wrapped * res_at_wrapped;
			}
		}

		__global__ void square_ker(float *ker, float *square_ker)
		{
			const int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index < %(DIM)s && area[index])
			{
				int temp = ker[index];
				square_ker[index] = temp * temp;
			}
		}

	
	"""
	code = code % {
		'DIM': dim,
		'BX': blockDimX,
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
	square_res_gpu = gpuarray.empty_like(res_gpu)
	square_ker_gpu = gpuarray.empty_like(ker_gpu)

	cuda.memcpy_htod(res_gpu, res_pin)
	cuda.memcpy_htod(ker_gpu, ker_pin)
	cuda.memcpy_htod(mdl_gpu, mdl_pin)
	cuda.memcpy_htod(area_gpu, area_pin)

	# Variables needed on the host side
	firstscore = -1
	score = -1
	best_score = -1
	argmax = 0
	max = 0

	# Compute the gain/phase of the kernel to start out
	h = cublasCreate()

	square_ker(ker_gpu, square_ker_gpu)
	ker_argmax = cublas.cublasIsamax(h, square_ker_gpu, square_ker_gpu.size, square_ker_gpu.gpudata, 1)
	q = 1/ker_pin[ker_max]

	n = 0
	while n < maxiter:

		nscore = 0
		step = gain * max * q

		mdl[argmax] += step

		compute_res(res_gpu, ker_gpu, step, square_res_gpu, area_gpu)

		nscore = cublas.cublasSasum(h, square_res_gpu.size, square_res_gpu.gpudata, 1)

		# Check for divergence
		nscore = sqrt(nscore/dim)
		if firstscore < 0:
			firstscore = score
		if score > 0 and nscore > score:
			if stop_if_div:
				# Diverged ---> so undo and quit

			elif best_score < 0 or score < best_score:
				# Diverged ---> buf prev score in case maximum

				# Reset maxiter counter
				n = 0

		elif score > 0 and (score - nscore) / firstscore < tol:
			# We are done
			cuda.memcpy_dtoh(res_pin, res_gpu)
			cuda.memcpy_dtoh(mdl_pin, mdl_gpu)
			return res_pin, res_gpu

		elif !stop_if_div and (best_score < 0 || nscore < best_score):
			# Reset maxiter counter
			n = 0

		# Find new max and update model
		nargmax = cublas.cublasIsamax(h, square_res_gpu.size, square_res_gpu.gpudata, 1)
		
		""" PROBABLY A FASTER WAY THAN THIS """
		cuda.memcpy_dtoh(res_pin, res_gpu)
		max = res_pin[nargmax]

		score = nscore
		argmax = nargmax
		n++


	cuda.memcpy_dtoh(res_pin, res_gpu)
	cuda.memcpy_dtoh(mdl_pin, mdl_gpu)
	return res_pin, res_gpu
