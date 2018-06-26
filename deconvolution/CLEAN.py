#!/usr/bin/env python

# maybe try pragma loop unrolls???

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 

def clean(res, ker, mdl=None, area=None, gain=0.1, maxiter=10000, tol=1e-3, stop_if_div=1, blockDimX=1024):
	s = cuda.Event()
	e = cuda.Event()

	gain = np.float32(gain)
	maxiter = np.int32(maxiter)
	tol = np.float32(tol)
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




	code = """
	#include <cuComplex.h>
	#include <cmath>
	__global__ void clean(float *res, float *ker, float *mdl, int *area, float tol, int stop_if_div)
	{
		float score = -1, nscore, best_score = -1;
		float max = 0, mmax, val, mval, step, q = 0, mq = 0;
		float firstscore = -1;
		int argmax = 0, nargmax = 0, wrap_n;
		float best_mdl[%(DIM)s], best_res[%(DIM)s];

		/* Compute gain/phase of kernel */
		for (int n = 0; n < %(DIM)s; n++)
		{
			val = ker[n];
			mval = val * val;
			if (mval > mq && area[n])
			{
				mq = mval;
				q = val;
			}
		}

		q = 1/q;
		/* The CLEAN loop */
		for (int i = 0; i < %(MAXITER)s; i++)
		{
			nscore = 0;
			mmax = -1;
			step = %(GAIN)s * max * q;
			mdl[argmax] += step;

			/* Take the next step and compute score */
			for (int n = 0; n < %(DIM)s; n++)
			{
				if (n + argmax >= %(DIM)s)
				{
					wrap_n = (n + argmax) - %(DIM)s;
				} else
				{
					wrap_n = (n + argmax);
				}
				res[wrap_n] -= ker[n] * step;
				val = res[wrap_n];
				mval = val * val;
				nscore += mval;
				if (mval > mmax && area[wrap_n])
				{
					nargmax = wrap_n;
					max = val;
					mmax = mval;
				}
			}

			nscore = sqrt(nscore / %(DIM)s);
			if (firstscore < 0)
			{
				firstscore = nscore;
			}
			score = nscore;
			argmax = nargmax;
		}
	}
	"""
	code = code % {
		'DIM': dim,
		'MAXITER': maxiter,
		'GAIN': gain,
	}

	mod = SourceModule(code)
	clean = mod.get_function("clean")

	res_pin = cuda.register_host_memory(res)
	ker_pin = cuda.register_host_memory(ker)
	mdl_pin = cuda.register_host_memory(mdl)
	area_pin = cuda.register_host_memory(area)

	res_gpu = cuda.mem_alloc(res.nbytes)
	ker_gpu = cuda.mem_alloc(ker.nbytes)
	mdl_gpu = cuda.mem_alloc(mdl.nbytes)
	area_gpu = cuda.mem_alloc(area.nbytes)


	cuda.memcpy_htod(res_gpu, res_pin)
	cuda.memcpy_htod(ker_gpu, ker_pin)
	cuda.memcpy_htod(mdl_gpu, mdl_pin)
	cuda.memcpy_htod(area_gpu, area_pin)


	clean.prepare("PPPPfi")
	clean.prepared_call((1,1,1), (1,1,1), res_gpu, ker_gpu, mdl_gpu, area_gpu, tol, stop_if_div)


	cuda.memcpy_dtoh(res_pin, res_gpu)

	return res_pin