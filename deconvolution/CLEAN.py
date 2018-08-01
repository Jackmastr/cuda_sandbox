#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import skcuda
from skcuda import cublas
import pycuda.gpuarray as gpuarray
from math import sqrt, ceil

def clean(res, ker, mdl=None, area=None, gain=0.1, maxiter=10000, tol=1e-3, stop_if_div=True, verbose=False):
	#s = cuda.Event()
	#e = cuda.Event()
	isComplex = (res.dtype == np.complex64)

	res = np.array(res)
	ker = np.array(ker)
	if mdl is not None:
		mdl = np.array(mdl)
	if area is not None:
		area = np.array(area)

	oneImg = (res.ndim == 1)
	gain = np.float64(gain)
	maxiter = np.int32(maxiter)
	tol = np.float64(tol)
	stop_if_div = np.int32(stop_if_div)

	# if isComplex:
	# 	if oneImg:
	# 		dim = len(res)
	# 		res = np.array([np.array([[np.real(r), np.imag(r)] for r in res], dtype=np.float32).flatten()])
	# 		ker = np.array([np.array([[np.real(k), np.imag(k)] for k in ker], dtype=np.float32).flatten()])

	# 		if mdl is None:
	# 			mdl = np.array([np.zeros(2*dim, dtype=np.float32)], dtype=np.float32)
	# 		else:
	# 			res[0]


	if oneImg:
		dim = len(res)
		res = np.array([res], dtype=np.float32)
		ker = np.array([ker], dtype=np.float32)

		if mdl is None:
			mdl = np.array([np.zeros(dim, dtype=np.float32)], dtype=np.float32)
		else:
			res[0] = res[0] - np.fft.ifft(np.fft.fft(mdl) * np.fft.fft(ker[0])).astype(np.float32)
			mdl = np.array([mdl], dtype=np.float32)

		if area is None:
			area = np.array([np.ones(dim, dtype=np.int32)], dtype=np.int32)
		else:
			area = np.array([area], dtype=np.int32)

	else:
		dim = len(res[0])
		numImgs = len(res)
		res = np.array(res, dtype=np.float32)

		if ker.ndim == 1:
			ker = np.array([ker]*numImgs, dtype=np.float32)
		else:
			ker = np.array(ker, dtype=np.float32)

		if mdl is None:
			mdl = np.array([np.zeros(dim, dtype=np.float32)]*numImgs, dtype=np.float32)
		elif mdl.ndim == 1:
			res = np.array([res[i] - np.fft.ifft(np.fft.fft(mdl) * np.fft.fft(ker[i])).astype(np.float32) for i in xrange(numImgs)])
			mdl = np.array([mdl]*numImgs, dtype=np.float32)
		else:
			res = np.array([res[i] - np.fft.ifft(np.fft.fft(mdl[i]) * np.fft.fft(ker[i])).astype(np.float32) for i in xrange(numImgs)])
			mdl = np.array(mdl, dtype=np.float32)

		if area is None:
			area = np.array([np.ones(dim, dtype=np.int32)]*numImgs, dtype=np.int32)
		elif area.ndim == 1:
			area = np.array([area]*numImgs, dtype=np.int32)
		else:
			area = np.array(area, dtype=np.int32)

	blockDimX = min(1024, len(ker))
	block = (blockDimX, 1, 1)

	grid = (int(ceil(len(ker)/blockDimX)), 1, 1)

	# block=(1,1,1)
	# grid=(1,1,1)

	# make all the arguments 1 level deeper of a pointer, use thread index to choose which one at the very start, then continue through like normal

	code_complex = """
	#pragma comment(linker, "/HEAP:40000000")
	#include <cuComplex.h>
	#include <stdio.h>
	#include <cmath>
	
	__global__ void cleanComplex(float *res, float *ker, float *mdl, int* area, int stop_if_div)
	{
		const int dim = %(DIM)s;
		const int maxiter = %(MAXITER)s;
		const double gain = %(GAIN)s;
		const double tol = %(TOL)s;
		const int index = blockDim.x * blockIdx.x + threadIdx.x;

		float *res = resP + index * 2 * %(DIM)s;
		float *ker = kerP + index * 2 * %(DIM)s;
		float *mdl = mdlP + index * 2 * %(DIM)s;
		int *area = areaP + index * 2 * %(DIM)s;


		float maxr=0, maxi=0, valr=0, vali, stepr, stepi, qr=0, qi=0;
		float score=-1, nscore, best_score=-1;
		float mmax, mval, mq=0;
		float firstscore=-1;
		int argmax=0, nargmax=0, wrap_n;
		
		float best_mdl[%(DIM)s * 2];
		float best_res[%(DIM)s * 2];
		
		// Compute gain/phase of kernel
		for (int n = 0; n < $(DIM)s; n++)
		{
			valr = ker[2 * n];
			vali = ker[2 * n + 1];
			mval = valr * valr + vali * vali;
			if (mval > mq && area[n])
			{
				mq = mval;
				qr = valr;
				qi = vali;
			}
		}
		qr /= mq;
		qi /= -mq;
		// The clean loop
		for (int i = 0; i < %(MAXITER)s; i++)
		{
			nscore = 0;
			mmax = -1;
			stepr = (float) gain * (maxr * qr - maxi * qi);
			stepi = (float) gain * (maxr * qi + maxi * qr);
			mdl[2 * argmax] += stepr;
			mdl[2 * argmax] += stepi;
			// Take next step and compute score
			for (int n = 0; n < %(DIM)s; n++)
			{
				wrap_n = (n + argmax) %% dim;
				res[2 * wrap_n] -= ker[2 * n] * stepr - ker[2 * n + 1] * stepi;
				res[2 * wrap_n + 1] -= ker[2 * n] * stepr + ker[2 * n + 1] * stepr;
				valr = res[2 * wrap_n];
				vali = res[2 * wrap_n + 1];
				mval = valr * valr + vali * vali;
				nscore += mval;
				if (mval > mmax && area[wrap_n])
				{
					nargmax = wrap_n;
					maxr = valr;
					maxi = vali;
					mmax = mval;
				}
			}
			nscore = sqrt(nscore/dim);
			if (firstscore < 0) firstscore = nscore;
			if (score > 0 && nscore > score)
			{
				if (stop_if_div)
				{
					// We've diverged: undo last step and give up
					mdl[2 * argmax] -= stepr;
					mdl[2 * argmax + 1] -= stepi;
					for (int n=0; n < dim; n++)
					{
						wrap_n = (n + argmax) %% dim;
						res[2 * wrap_n] += ker[2 * n] * stepr - ker[2 * n + 1] * stepi;
						res[2 * wrap_n + 1] += ker[2 * n] * stepi + ker[2 * n + 1] * stepr;
					}
					return;
				} else if (best_score < 0 || score < best_score)
				{
					// We've diverged: buf prev score in case it's global best
					for (int n=0; n < dim; n++)
					{
						wrap_n = (n + argmax) %% dim;
						best_mdl[2 * n] = mdl[2 * n];
						best_mdl[2 * n + 1] = mdl[2 * n + 1];
						best_res[2 * wrap_n] = res[2 * wrap_n] + ker[2 * n] * stepr - ker[2 * n + 1] * stepi;
						best_res[2 * wrap_n + 1] = res[2 * wrap_n + 1] + ker[2 * n] * stepi + ker[2 * n + 1] * stepr;
					}
					best_mdl[2 * argmax] -= stepr;
					best_mdl[2 * argmax + 1] -= stepi;
					best_score = score;
					i = 0; // Reset maxiter counter
				}
			} else if (score > 0 && (score - nscore) / firstscore < tol)
			{
				// We're done
				return;
			} else if (not stop_if_div && (best_score < 0 || nscore < best_score))
			{
				i = 0; // Reset maxiter counter
			}
			score = nscore;
			argmax = nargmax;
		}
		// If we end on maxiter, then make sure mdl/res reflect best score
		if (best_score > 0 && best_score < nscore)
		{
			for (int n=0; n < dim; n++)
			{
				mdl[2 * n] = best_mdl[2 * n];
				mdl[2 * n + 1] = best_mdl[2 * n + 1];
				res[2 * n] = best_res[2 * n];
				res[2 * n + 1] = best_res[2 * n + 1];
			}
		}
	"""
	
	
	code = """
	#pragma comment(linker, "/HEAP:40000000")

	#include <cuComplex.h>
	#include <stdio.h>
	#include <cmath>

	__global__ void clean(float *resP, float *kerP, float *mdlP, int *areaP, int stop_if_div)
	{
		const int dim = %(DIM)s;
		const int maxiter = %(MAXITER)s;
		const double gain = %(GAIN)s;
		const double tol = %(TOL)s;
		const int index = blockDim.x * blockIdx.x + threadIdx.x;

		float *res = resP + index * %(DIM)s;
		float *ker = kerP + index * %(DIM)s;
		float *mdl = mdlP + index * %(DIM)s;
		int *area = areaP + index * %(DIM)s;

		float score=-1, nscore, best_score=-1;
       	float max=0, mmax, val, mval, step, q=0, mq=0;
        float firstscore=-1;
        int argmax=0, nargmax=0, wrap_n;

        float best_mdl[%(DIM)s], best_res[%(DIM)s];

        // Compute gain/phase of kernel
        for (int n=0; n < dim; n++) {
            val = *(ker + n);
            mval = val * val;
            if (mval > mq && *(area + n)) {
                mq = mval;
                q = val;
            }
        }
        q = 1/q;
        // The clean loop
        for (int i=0; i < maxiter; i++) {
            nscore = 0;
            mmax = -1;
            step =  gain * max * q;
            *(mdl + argmax) += step;
            // Take next step and compute score
            for (int n=0; n < dim; n++) {
                wrap_n = (n + argmax) %% dim;
                *(res + wrap_n) -= *(ker + n) * step;
                val = *(res + wrap_n);
                mval = val * val;
                nscore += mval;
                if (mval > mmax && *(area + wrap_n)) {
                    nargmax = wrap_n;
                    max = val;
                    mmax = mval;
                }
            }
            nscore = sqrt(nscore / dim);
            if (firstscore < 0) firstscore = nscore;

			//printf("MY CLEAN Iter %%d: Max=(%%d), Score = %%f, Prev = %%f\\n", \
            //        i, nargmax, (double) (nscore/firstscore), \
			//		(double) (score/firstscore));

            if (score > 0 && nscore > score) {
                if (stop_if_div) {
                    // We've diverged: undo last step and give up
                    *(mdl + argmax) -= step;
                    for (int n=0; n < dim; n++) {
                        wrap_n = (n + argmax) %% dim;
                        *(res + wrap_n) += *(ker + n) * step;
                    }
                    return;
                } else if (best_score < 0 || score < best_score) {
                    // We've diverged: buf prev score in case it's global best
                    for (int n=0; n < dim; n++) {
                        wrap_n = (n + argmax) %% dim;
                        best_mdl[n] = *(mdl + n);
                        best_res[wrap_n] = *(res + wrap_n) + *(ker + n) * step;
                    }
                    best_mdl[argmax] -= step;
                    best_score = score;
                    i = 0;  // Reset maxiter counter
                }
            } else if (score > 0 && (score - nscore) / firstscore < tol) {
                // We're done
                return;
            } else if (!stop_if_div && (best_score < 0 || nscore < best_score)) {
                i = 0;  // Reset maxiter counter
            }
            score = nscore;
            argmax = nargmax;
        }
        // If we end on maxiter, then make sure mdl/res reflect best score
        if (best_score > 0 && best_score < nscore) {
            for (int n=0; n < dim; n++) {
                *(mdl + n) = best_mdl[n];
                *(res + n) = best_res[n];
           }
        }
	}
	
	"""
	code = code % {
		'DIM': dim,
		'MAXITER': maxiter,
		'GAIN': gain,
		'TOL': tol,
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

	clean.prepare("PPPPi")
	clean.prepared_call(grid, block, res_gpu, ker_gpu, mdl_gpu, area_gpu, stop_if_div)

	cuda.memcpy_dtoh(res_pin, res_gpu)
	cuda.memcpy_dtoh(mdl_pin, mdl_gpu)

	if oneImg:
		return mdl_pin[0], res_pin[0]
	return mdl_pin, res_pin
