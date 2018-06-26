#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 

def clean(res, ker, mdl, area, gain, maxiter, tol, stop_if_div, blockDimX=1024)
	s = cuda.Event()
	e = cuda.Event()

	code = """
	#include <cuComplex.h>
	#include <cmath>
	__global__ void clean(float *res, float *ker, float gain, int maxiter, float tol, int stop_if_div, int dim)
	{
		float mdl[dim]

		// mxxx is the index where xxx occurs?

		int score = -1, nscore, best_score = -1;
		float max = 0, mmax, val, mval, step, q = 0, mq = 0;
		float firstscore = -1;
		int argmax = 0, nargmax = 0, wrap_n;

		float best_md1[dim], best_res[dim];

		// Compute gain/phase of kernel
		for (int n = 0; n < dim; n++)
		{
			val = ker[n];
			mval = val * val;
			if (mval > mq)
			{
				mq = mval;
				q = val;
			}
		}

		q = 1/q;

		// The clean loop
		for (int i = 0; i < maxiter; i++)
		{
			nscore = 0;
			nmax = -1;
			step = gain * max * q;
			md1[argmax] += step;

			// Take the next step and compute score
			for (int n = 0; n < dim; n++)
			{
				wrap_n = (n + argmax) %% dim;
				res[wrap_n] -= ker[n] * step;
				val = res[wrap_n];
				mval = val * val;
				nscore += mval;
				if (mval > mmax)
				{
					nargmax = wrap_n;
					max = val;
					mmax = mval;
				}
			}
		}

	}
	"""
	mod = SourceModule(code)
	clean = mod.get_function("clean")

	gain = np.float32(gain)
	maxiter = np.int32(maxiter)
	tol = np.float32(tol)
	stop_if_div = np.int32(stop_if_div)

	res = np.array(res, dtype=np.float32)
	ker = np.array(ker, dtype=np.float32)

	dim = np.int32(len(res))

	res_pin = cuda.register_host_memory(res)
	ker_pin = cuda.register_host_memory(ker)

	res_gpu = cuda.mem_alloc(res.nbytes)
	ker_gpu = cuda.mem_alloc(ker.nbytes)

	cuda.memcpy_htod(res_gpu, res_pin)
	cuda.memcpy_htod(ker_gpu, ker_pin)


	clean.prepare("PPfifii")
	clean.prepared_call(res_gpu, ker_gpu, gain, maxiter, tol, stop_if_div, dim)


	cuda.memcpy_dtoh(res_pin, res_gpu)

	return res_pin