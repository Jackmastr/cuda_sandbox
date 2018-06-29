#!/usr/bin/env python

# maybe try pragma loop unrolls???

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 

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

	block=(1,1,1)
	grid=(1,1,1)

	code = """
	#include <cuComplex.h>
	#include <cmath>
	#include <stdio.h>

	__device__ void take_step(float *res, float *ker, int n, int argmax, int step)
	{
		int index = threadIdx.x;
		if (index < %(DIM)s)
		{
			int wrap = (index + argmax) %% %(DIM)s;
			*(res + wrap) -= *(ker + n) * step;
		}
	}

	__global__ void clean(float *res, float *ker, float *mdl, int *area, double tol, int stop_if_div)
	{
		const int dim = %(DIM)s;
		const int maxiter = %(MAXITER)s;
		const double gain = %(GAIN)s;

		float score=-1, nscore, best_score=-1;
       	float max=0, mmax, val, mval, step, q=0, mq=0;
        float firstscore=-1;
        int argmax=0, nargmax=0, wrap_n;
        float *best_mdl=NULL, *best_res=NULL;
        if (!stop_if_div) {
            best_mdl = (float *)malloc(dim*sizeof(float));
            best_res = (float *)malloc(dim*sizeof(float));
        }
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
            step = (float) gain * max * q;
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
            if (score > 0 && nscore > score) {
                if (stop_if_div) {
                    // We've diverged: undo last step and give up
                    *(mdl + argmax) -= step;
                    for (int n=0; n < dim; n++) {
                        wrap_n = (n + argmax) %% dim;
                        *(res + wrap_n) += *(ker + n) * step;
                    }
                    printf("stopped at iter: %%d\\n", i);
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
                if (best_mdl != NULL) { free(best_mdl); free(best_res); }
                printf("stopped at iter: %%d\\n", i);
                return;
            } else if (not stop_if_div && (best_score < 0 || nscore < best_score)) {
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
        if (best_mdl != NULL) { free(best_mdl); free(best_res); }
        printf("stopped at iter: %%d\\n", %(MAXITER)s);
        return;
	}
	"""
	code = code % {
		'DIM': dim,
		'MAXITER': maxiter,
		'GAIN': gain,
	}

	# s.record()
	mod = SourceModule(code)
	clean = mod.get_function("clean")
	# e.record()
	# e.synchronize()
	#print "Compilation:", s.time_till(e), "ms"

	res_pin = cuda.register_host_memory(res)
	ker_pin = cuda.register_host_memory(ker)
	mdl_pin = cuda.register_host_memory(mdl)
	area_pin = cuda.register_host_memory(area)

	res_gpu = cuda.mem_alloc(res.nbytes)
	ker_gpu = cuda.mem_alloc(ker.nbytes)
	mdl_gpu = cuda.mem_alloc(mdl.nbytes)
	area_gpu = cuda.mem_alloc(area.nbytes)

	# s.record()
	cuda.memcpy_htod(res_gpu, res_pin)
	cuda.memcpy_htod(ker_gpu, ker_pin)
	cuda.memcpy_htod(mdl_gpu, mdl_pin)
	cuda.memcpy_htod(area_gpu, area_pin)
	# e.record()
	# e.synchronize()
	#print "transfer host to device:", s.time_till(e), "ms"

	s.record()
	clean.prepare("PPPPdi")
	clean.prepared_call(grid, block, res_gpu, ker_gpu, mdl_gpu, area_gpu, tol, stop_if_div)
	e.record()


	cuda.memcpy_dtoh(mdl_pin, mdl_gpu)
	print "kernel execution:", s.time_till(e), "ms"


	return mdl_pin