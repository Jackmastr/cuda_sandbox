#!/usr/bin/env python

# Eventually get this to work over multiple GPUs, hopefully

# Test without using so many format stream tricks, it may be making it slower

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps


def MedianFilter(input=None, kernel_size=3, bw=16, bh=16, n=256, m=0, timing=False, nStreams=0):

	BLOCK_WIDTH = bw
	BLOCK_HEIGHT = bh

	WINDOW_SIZE = kernel_size

	if isinstance(kernel_size, (int, long)):
		kernel_size = [kernel_size]*2

	WS_x, WS_y = kernel_size
	padding_y = WS_x/2
	padding_x = WS_y/2

	N = np.int32(n)
	if m == 0:
		M = np.int32(n)
	else:
		M = np.int32(m)

	#indata = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)
	if input is None:
		indata = np.array(np.random.rand(M, N), dtype=np.float32)
	else:
		indata = np.array(input, dtype=np.float32)

	s = cuda.Event()
	e = cuda.Event()
	s.record()


	expanded_N = N + (2 * padding_y)
	expanded_M = M + (2 * padding_x)

	gridx = max(1, int(np.ceil((expanded_N)/BLOCK_WIDTH))+1)
	gridy = max(1, int(np.ceil((expanded_M)/BLOCK_HEIGHT))+1)
	grid = (gridx,gridy)
	block = (BLOCK_WIDTH, BLOCK_HEIGHT, 1)


	code = """
		#include <stdio.h>

		__global__ void mf(float* in, float* out, int imgWidth, int imgHeight, int ws)
		{
			int ws_sq = ws * ws;
			float window[81];

			const int x_thread_offset = 16 * blockIdx.x + threadIdx.x;
			const int y_thread_offset = 16 * blockIdx.y + threadIdx.y;

			for (int y = ws/2 + y_thread_offset; y < imgHeight - ws/2; y += 16 * gridDim.y)
			{
				for (int x = ws/2 + x_thread_offset; x < imgWidth - ws/2; x += 16 * gridDim.x)
				{
					int i = 0;
					for (int fx = 0; fx < ws; ++fx)
					{
						for (int fy = 0; fy < ws; ++fy)
						{
							window[i] = in[(x + fx - ws/2) + (y + fy - ws/2)*imgWidth];
							i += 1;
						}
					}

					// Sort to find the median
					for (int j = 0; j < ws_sq; ++j)
					{
						for (int k = j + 1; k < ws_sq; ++k)
						{
							if (window[j] > window[k])
							{
								float tmp = window[j];
								window[j] = window[k];
								window[k] = tmp;
							}
						}
					}
					out[y*imgWidth + x] = window[ws_sq/2];
				}
			}
		}

		"""
	# s.record()
	mod = SourceModule(code)
	mf = mod.get_function('mf')
	e.record()
	# e.synchronize()
	# print s.time_till(e), "ms"

	indata = np.pad(indata, ( (padding_y, padding_y), (padding_x, padding_x) ), 'constant', constant_values=0)
	outdata = np.empty_like(indata)



	in_pin = cuda.register_host_memory(indata)
	out_pin = cuda.register_host_memory(outdata)

	in_gpu = cuda.mem_alloc(indata.nbytes)
	out_gpu = cuda.mem_alloc(outdata.nbytes)

	# s.record()
	cuda.memcpy_htod_async(in_gpu, in_pin)
	# e.record()
	# e.synchronize()
	# print s.time_till(e), "ms"


	# s.record()
	mf.prepare("PPii")
	mf.prepared_call(grid, block, in_gpu, out_gpu, expanded_M, expanded_N, WINDOW_SIZE)
	# e.record()
	# e.synchronize()
	# print s.time_till(e), "ms"


	cuda.memcpy_dtoh(out_pin, out_gpu)


	if (nStreams > 0 and N > nStreams):
		N = expanded_N/nStreams
		N_lo = expanded_N % nStreams # leftover if N doesn't divide evenly into the streams

		stream = []
		in_pin_list = []
		outdata_list = []
		in_gpu_list = []
		out_gpu_list = []
		for i in xrange(nStreams):
			stream.append(cuda.Stream())

			if (i < nStreams - 1):
				a = indata[i*N:(i+1)*N + WINDOW_SIZE/2]
				in_pin_list.append(a)
				outdata_list.append(np.empty_like(a))

			else:
				a = indata[i*N:]
				in_pin_list.append(a)
				outdata_list.append(np.empty_like(a))

			in_gpu_list.append(cuda.mem_alloc(in_pin_list[i].nbytes))
			out_gpu_list.append(cuda.mem_alloc(outdata_list[i].nbytes))

		for i in xrange(nStreams + 2):
			ii = i - 1
			iii = i - 2

			if 0 <= iii < nStreams:
				st = stream[iii]
				cuda.memcpy_dtoh_async(in_pin_list[iii], in_gpu_list[iii], stream=st)
				cuda.memcpy_htod_async(out_gpu_list[iii], outdata_list[iii], stream=st)

			if 0 <= ii < nStreams:
				st = stream[ii]
				if ii < nStreams - 1:
					mf(in_gpu_list[ii], outdata_list[ii], expanded_M, N, grid=grid, block=block, stream=st)
				else:
					mf(in_gpu_list[ii], outdata_list[ii], expanded_M, N + N_lo, grid=grid, block=block, stream=st)

			if 0 <= i < nStreams:
				st = stream[i]
				cuda.memcpy_htod_async(in_gpu_list[i], in_pin_list[i], stream=st)
				
		print outdata_list
		#outdata = np.concatenate(outdata_list, axis=1)




	if (padding_y > 0):
		outdata = outdata[padding_y:-padding_y]
	if (padding_x > 0):
		outdata = outdata[:, padding_x:-padding_x]

	if (timing):
		e.record()
		e.synchronize()
		print "THIS FUNCTION: ", s.time_till(e), "ms"


		s.record()
		true_ans= sps.medfilt2d(indata, (WS_x, WS_y))
		e.record()
		e.synchronize()
		print "SCIPY MEDFILT", s.time_till(e), "ms"

	return outdata

	# return np.allclose(out_pin, true_ans)
	# print out_pin
	# print true_ans