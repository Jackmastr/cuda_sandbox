#!/usr/bin/env python

# maybe try pragma loop unrolls???

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
	oneImg = False

	gain = np.float64(gain)
	maxiter = np.int32(maxiter)
	tol = np.float64(tol)
	stop_if_div = np.int32(stop_if_div)


	res = np.array(res, dtype=np.float32)
	if res.ndim == 1:
		oneImg = True
		res = np.array([res], dtype=np.float32)

	ker = np.array(ker, dtype=np.float32)
	if ker.ndim == 1:
		ker = np.array([ker], dtype=np.float32)

	dim = np.int32(len(res[0]))

	if mdl is None:
		mdl = np.array([np.zeros(dim)]*len(ker), dtype=np.float32)
	else:
		mdl = np.array(mdl, dtype=np.float32)

	if mdl.ndim == 1:
		mdl = np.array([mdl], dtype=np.float32)
	
	if area is None:
		area = np.array([np.ones(dim)]*len(ker), dtype=np.int32)
	else:
		area = np.array(area, dtype=np.int32)

	if area.ndim == 1:
		area = np.array([area], dtype=np.int32)

	blockDimX = min(1024, len(ker))
	block = (blockDimX, 1, 1)

	grid = (int(ceil(len(ker)/blockDimX)), 1, 1)

	# block=(1,1,1)
	# grid=(1,1,1)

	# make all the arguments 1 level deeper of a pointer, use thread index to choose which one at the very start, then continue through like normal

	code = """
	#pragma comment(linker, "/HEAP:40000000")

	#include <cuComplex.h>
	#include <stdio.h>


	__global__ cleanC(float *resP, float *kerP, float *mdlP, int *areaP, double tol, int stop_if_div)
	{
        T maxr=0, maxi=0, valr, vali, stepr, stepi, qr=0, qi=0;
        T score=-1, nscore, best_score=-1;
        T mmax, mval, mq=0;
        T firstscore=-1;
        int argmax=0, nargmax=0, dim=DIM(res,0), wrap_n;
        T *best_mdl=NULL, *best_res=NULL;
        if (!stop_if_div) {
            best_mdl = (T *)malloc(2*dim*sizeof(T));
            best_res = (T *)malloc(2*dim*sizeof(T));
        }
        // Compute gain/phase of kernel
        for (int n=0; n < dim; n++) {
            valr = CIND1R(ker,n,T);
            vali = CIND1I(ker,n,T);
            mval = valr * valr + vali * vali;
            if (mval > mq && IND1(area,n,int)) {
                mq = mval;
                qr = valr;
                qi = vali;
            }
        }
        qr /= mq;
        qi = -qi / mq;
        // The clean loop
        for (int i=0; i < maxiter; i++) {
            nscore = 0;
            mmax = -1;
            stepr = (T) gain * (maxr * qr - maxi * qi);
            stepi = (T) gain * (maxr * qi + maxi * qr);
            CIND1R(mdl,argmax,T) += stepr;
            CIND1I(mdl,argmax,T) += stepi;
            // Take next step and compute score
            for (int n=0; n < dim; n++) {
                wrap_n = (n + argmax) % dim;
                CIND1R(res,wrap_n,T) -= CIND1R(ker,n,T) * stepr - \
                                        CIND1I(ker,n,T) * stepi;
                CIND1I(res,wrap_n,T) -= CIND1R(ker,n,T) * stepi + \
                                        CIND1I(ker,n,T) * stepr;
                valr = CIND1R(res,wrap_n,T);
                vali = CIND1I(res,wrap_n,T);
                mval = valr * valr + vali * vali;
                nscore += mval;
                if (mval > mmax && IND1(area,wrap_n,int)) {
                    nargmax = wrap_n;
                    maxr = valr;
                    maxi = vali;
                    mmax = mval;
                }
            }
            nscore = sqrt(nscore / dim);
            if (firstscore < 0) firstscore = nscore;
            if (verb != 0)
                printf("Iter %d: Max=(%d), Score = %f, Prev = %f\n", \
                    i, nargmax, (double) (nscore/firstscore), \
                    (double) (score/firstscore));
            if (score > 0 && nscore > score) {
                if (stop_if_div) {
                    // We've diverged: undo last step and give up
                    CIND1R(mdl,argmax,T) -= stepr;
                    CIND1I(mdl,argmax,T) -= stepi;
                    for (int n=0; n < dim; n++) {
                        wrap_n = (n + argmax) % dim;
                        CIND1R(res,wrap_n,T) += CIND1R(ker,n,T) * stepr - CIND1I(ker,n,T) * stepi;
                        CIND1I(res,wrap_n,T) += CIND1R(ker,n,T) * stepi + CIND1I(ker,n,T) * stepr;
                    }
                    return -i;
                } else if (best_score < 0 || score < best_score) {
                    // We've diverged: buf prev score in case it's global best
                    for (int n=0; n < dim; n++) {
                        wrap_n = (n + argmax) % dim;
                        best_mdl[2*n+0] = CIND1R(mdl,n,T);
                        best_mdl[2*n+1] = CIND1I(mdl,n,T);
                        best_res[2*wrap_n+0] = CIND1R(res,wrap_n,T) + CIND1R(ker,n,T) * stepr - CIND1I(ker,n,T) * stepi;
                        best_res[2*wrap_n+1] = CIND1I(res,wrap_n,T) + CIND1R(ker,n,T) * stepi + CIND1I(ker,n,T) * stepr;
                    }
                    best_mdl[2*argmax+0] -= stepr;
                    best_mdl[2*argmax+1] -= stepi;
                    best_score = score;
                    i = 0;  // Reset maxiter counter
                }
            } else if (score > 0 && (score - nscore) / firstscore < tol) {
                // We're done
                if (best_mdl != NULL) { free(best_mdl); free(best_res); }
                return i;
            } else if (not stop_if_div && (best_score < 0 || nscore < best_score)) {
                i = 0;  // Reset maxiter counter
            }
            score = nscore;
            argmax = nargmax;
        }
        // If we end on maxiter, then make sure mdl/res reflect best score
        if (best_score > 0 && best_score < nscore) {
            for (int n=0; n < dim; n++) {
                CIND1R(mdl,n,T) = best_mdl[2*n+0];
                CIND1I(mdl,n,T) = best_mdl[2*n+1];
                CIND1R(res,n,T) = best_res[2*n+0];
                CIND1I(res,n,T) = best_res[2*n+1];
            }
        }   
        if (best_mdl != NULL) { free(best_mdl); free(best_res); }
        return maxiter;
	}










	__global__ void clean(float *resP, float *kerP, float *mdlP, int *areaP, double tol, int stop_if_div)
	{
		const int dim = %(DIM)s;
		const int maxiter = %(MAXITER)s;
		const double gain = %(GAIN)s;
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

	clean.prepare("PPPPdi")
	clean.prepared_call(grid, block, res_gpu, ker_gpu, mdl_gpu, area_gpu, tol, stop_if_div)

	cuda.memcpy_dtoh(res_pin, res_gpu)
	cuda.memcpy_dtoh(mdl_pin, mdl_gpu)

	if oneImg:
		return mdl_pin[0], res_pin[0]
	return mdl_pin, res_pin
