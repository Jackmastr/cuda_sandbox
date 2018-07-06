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

def clean(res, ker, mdl=None, area=None, gain=0.1, maxiter=10000, tol=1e-3, stop_if_div=True, verbose=False, blockDimX=1024):
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

		texture<float, 1> tex_ker;

		__device__ void compute_res(float *res, float *ker, float step, float *square_res, int* area, int argmax)
		{
			const int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index < %(DIM)s && area[index])
			{
				int wrapped_index = (index + argmax) %% %(DIM)s;
				res[wrapped_index] -= ker[index] * step;
				float res_at_wrapped = res[wrapped_index];
				// ASK ABOUT 'AREA' IN THIS CASE
				square_res[wrapped_index] = res_at_wrapped * res_at_wrapped;
			}
		}

		__global__ void doNothing(float *res, float *ker, float step, float *square_res, int* area, int argmax)
		{
			// just to compare how long it takes
		}

		__device__ void square_ker(float *ker, float *square_ker, int *area)
		{
			const int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index < %(DIM)s && area[index])
			{
				float temp = ker[index];
				square_ker[index] = temp * temp;
			}
		}

		__device__ void bufBest(float *res, float *best_res, float *ker, float *mdl, float *best_mdl, int *area, float step, int argmax)
		{
			const int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index < %(DIM)s && area[index])
			{
				int wrapped_index = (index + argmax) %% %(DIM)s;
				best_res[wrapped_index] = res[wrapped_index] + ker[index] * step;
				best_mdl[index] = mdl[index];

				if (index == 0)
				{
					best_mdl[argmax] -= step;
				}
			}
		}

		__global__ void doEverything(float *res, float *ker, float step, float *square_res, int* area, int argmax)
		{
			const int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index >= %(DIM)s)
			{
				return;
			}

			for (int n = 0; n < %(MAXITER)s; n++)
			{
				compute_res(res, ker, step, square_res, area, argmax);



				//square_ker(ker, square_ker, area);

				bufMaxRes(res, res, ker, area, step, argmax);

				if (index == 0)
				{
					ker[0] += res[0];
				}

				__syncthreads();

				compute_res(res, ker, step, square_res, area, argmax);

				if (index == 0)
				{
					ker[0] -= res[0];
				}

				__syncthreads();

			}
		}

		__device__ int getArgmax(float *arr)
		{
			return 0;
		}

		__device__ float sum(float *arr)
		{
			return 0;
		}

		__device__ void undoStep(float *res, float *ker, int step, int *area, int argmax)
		{
			return;
		}

		__device__ void overwrite(float *old, float *new)
		{
			const int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index >= %(DIM)s)
			{
				return;
			}
			// can you just move the orig pointer????
			old[index] = new[index];
		}

		__global__ void clean(float *res, float *ker, float *mdl, int *area)
		{
			const int index = threadIdx.x + blockDim.x * blockIdx.x;

			float score = -1, nscore, best_score = -1;
			float max = 0, mmax, step, q = 0;
			float firstscore = -1;
			int argmax = 0, nargmax = 0;

			float best_mdl[%(DIM)s];
			float best_res[%(DIM)s];
			float ker_squared[%(DIM)s];
			float res_squared[%(DIM)s];

			/* Compute gain / phase of the kernel */
			square_ker(ker, ker_squared, area);
			q = ker[ getArgmax(ker_squared) ];
			q = 1/q;

			/* The clean loop */
			for (int i = 0; i < %(MAXITER)s; i++)
			{
				nscore = 0;
				mmax = -1;
				step = (float) %(GAIN)s * max * q;

				if (index == 0)
				{
					mdl[argmax] += step;
				}
				__syncthreads();

				/* Take the next step and compute score */
				compute_res(res, ker, step, res_squared, area, argmax);

				__syncthreads(); // Not sure if this is needed

				max = res[ getArgmax(res_squared) ];
				nscore = sum(res_squared);

				__syncthreads(); // Not sure if this is needed

				if (index == 0)
				{
					nscore = sqrt(nscore/%(DIM)s);
				}

				__synchthreads();

				if (firstscore < 0)
				{
					firstscore = nscore;
				}

				if (score > 0 && nscore > score)
				{
					if (%(STOPIFDIV)s)
					{
						if (index == 0)
						{
							mdl[argmax] -= step;
						}
						__syncthreads();

						undoStep(res, ker, step, area, argmax)

						return;
					}
					else if (best_score < 0 || score < best_score)
					{
						/* We've diverged so buf prev score in case global best */
						bufMaxRes(res, best_res, ker, mdl, best_mdl, area, step, argmax);
						__syncthreads();
						best_score = score;
						i = 0;
					}

				}
				else if (score > 0 && (score - nscore) / firstscore < tol)
				{
					/* We're done! */
					return;
				}
				else if (%(STOPIFDIV)s == 0 && (best_score < 0 || nscore < best_score)
				{
					/* Reset maxiter counter */
					i = 0;
				}

				score = nscore;
				argmax = nargmax;

			}
			/* If we end on maxiter, then make sure mdl/res reflect best_score
			if (best_score > 0 && best_score < nscore)
			{
				overwrite(mdl, best_mdl);
				overwrite(res, best_res);
			}

		}


	
	"""
	code = code % {
		'DIM': dim,
		'BX': blockDimX,
		'MAXITER': maxiter,
		'GAIN': gain,
		'STOPIFDIV': int(stop_if_div)
	}
	mod = SourceModule(code)
	#tex_ker = mod.get_texref("tex_ker")
	#cuda.matrix_to_texref(np.array([ker]), tex_ker, order="C")

	# compute_res = mod.get_function("compute_res")
	# square_ker = mod.get_function("square_ker")
	# bufMaxRes = mod.get_function("bufMaxRes")
	# doNothing = mod.get_function("doNothing")
	doEverything = mod.get_function("doEverything")

	res_pin = cuda.register_host_memory(res)
	ker_pin = cuda.register_host_memory(ker)
	area_pin = cuda.register_host_memory(area)

	res_gpu = cuda.mem_alloc(res.nbytes)
	ker_gpu = cuda.mem_alloc(ker.nbytes)
	area_gpu = cuda.mem_alloc(area.nbytes)

	# Additional buffers that may be needed
	square_res_gpu = gpuarray.empty(res.shape, np.float32)
	square_ker_gpu = gpuarray.empty(ker.shape, np.float32)
	best_res_gpu = cuda.mem_alloc(res.nbytes)

	cuda.memcpy_htod(res_gpu, res_pin)
	cuda.memcpy_htod(ker_gpu, ker_pin)
	cuda.memcpy_htod(area_gpu, area_pin)


	# Variables needed on the host side
	firstscore = -1
	score = -1
	best_score = -1
	argmax = np.int32(0)
	max = 0
	best_mdl = None

	tot_timeCR = 0
	tot_timeDN = 0

	# Compute the gain/phase of the kernel to start out
	h = cublas.cublasCreate()

	# square_ker(ker_gpu, square_ker_gpu, area_gpu, block=block, grid=grid)
	# ker_argmax = cublas.cublasIsamax(h, square_ker_gpu.size, square_ker_gpu.gpudata, 1)
	# cuda.memcpy_dtoh(ker_pin, ker_gpu)
	# ker_max = ker_pin[ker_argmax]
	# q = 1/ker_max

	# n = 0

	# compute_res.prepare("PPfPPi")
	# doNothing.prepare("PPfPPi")

	step = 0.1

	s.record()
	doEverything.prepare("PPfPPi")
	doEverything.prepared_call(grid, block, res_gpu, ker_gpu, step, square_res_gpu.gpudata, area_gpu, argmax)
	e.record()
	e.synchronize()
	print "DO EVERYTHING TAKES:", s.time_till(e), "ms"

	return;

	while n < maxiter:

		nscore = 0
		step = np.float32(gain * max * q)

		mdl[argmax] += step

		# s.record()
		compute_res.prepared_call(grid, block, res_gpu, ker_gpu, step, square_res_gpu.gpudata, area_gpu, argmax)
		# e.record()
		# e.synchronize()
		# tot_timeCR += s.time_till(e)

		# s.record()
		doNothing.prepared_call(block, grid, res_gpu, ker_gpu, step, square_res_gpu.gpudata, area_gpu, argmax)
		# e.record()
		# e.synchronize()
		# tot_timeDN += s.time_till(e)

		nscore = cublas.cublasSasum(h, square_res_gpu.size, square_res_gpu.gpudata, 1)

		#print "square res", square_res_gpu.get()

		# if verbose:
			# print "Iter %d: Max=(%d), Score = %f, Prev= %f\n", i, nargmax, float(nscore/firstscore), float(score/firstscore)


		# Check for divergence
		nscore = sqrt(float(nscore/dim))
		if firstscore < 0:
			firstscore = nscore
		if score > 0 and nscore > score:
			if stop_if_div:
				# Diverged ---> so undo and quit
				mdl[argmax] -= step
				compute_res(res_gpu, -step, square_res_gpu, area_gpu, argmax, block=block, grid=grid)
				#print "stopped because it diverged"
				break

			elif best_score < 0 or score < best_score:
				# Diverged ---> buf prev score in case maximum
				best_mdl = list(mdl)
				bufMaxRes(res_gpu, best_res_gpu, ker_gpu, area_gpu, step, argmax, block=block, grid=grid)

				best_mdl[argmax] -= step
				best_score = score
				# Reset maxiter counter 
				n = 0

		elif score > 0 and float(score - nscore) / firstscore < tol:
			# We are done
			# print "score", score, "nscore", nscore
			#print "stopped because within tolerance", float(score - nscore) / firstscore, "<", tol
			break

		elif not stop_if_div and (best_score < 0 or nscore < best_score):
			# Reset maxiter counter
			n = 0

		# Find new max and update model
		nargmax = cublas.cublasIsamax(h, square_res_gpu.size, square_res_gpu.gpudata, 1)

		
		""" PROBABLY A FASTER WAY THAN THIS """

		cuda.memcpy_dtoh(res_pin, res_gpu)



		max = res_pin[nargmax]

		score = nscore
		argmax = np.int32(nargmax)

		n += 1


	# print "EVERY DONOTHING:", tot_timeDN, "ms"
	# print "EVERY COMPUTE_RES:", tot_timeCR, "ms"
	# print "DIF =", tot_timeCR - tot_timeDN, "ms"

	if best_score > 0 and best_score < nscore:
		cuda.memcpy_dtoh(res_pin, best_res_gpu)
		mdl = best_mdl
	else:
		cuda.memcpy_dtoh(res_pin, res_gpu)

	return mdl, res_pin
