#!/usr/bin/env python

# Eventually get this to work over multiple GPUs, hopefully
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps


def MedianFilter(indata=None, ws=3, bw=16, bh=16, n=256, m=0, timing=False, nStreams=0):

	BLOCK_WIDTH = bw
	BLOCK_HEIGHT = bh

	WINDOW_SIZE = ws

	padding = WINDOW_SIZE/2

	N = np.int32(n)
	if m == 0:
		M = np.int32(n)
	else:
		M = np.int32(m)

	#indata = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)
	if indata is None:
		indata = np.array(np.random.rand(M, N), dtype=np.float32)
	else:
		indata = np.array(indata, dtype=np.float32)

	s = cuda.Event()
	e = cuda.Event()
	s.record()


	expanded_N = N + (2 * padding)
	expanded_M = M + (2 * padding)

	gridx = max(1, int(np.ceil((expanded_N)/BLOCK_WIDTH)))
	gridy = max(1, int(np.ceil((expanded_M)/BLOCK_HEIGHT)))
	grid = (gridx,gridy)
	block = (BLOCK_WIDTH, BLOCK_HEIGHT, 1)


	code = """
		#include <stdio.h>

		__global__ void mf(float* in, float* out, int imgWidth, int imgHeight)
		{

			float window[%(WS^2)s];

			const int x_thread_offset = %(BW)s * blockIdx.x + threadIdx.x;
			const int y_thread_offset = %(BH)s * blockIdx.y + threadIdx.y;

			for (int y = %(WS/2)s + y_thread_offset; y < imgHeight - %(WS/2)s; y += %(y_stride)s)
			{
				for (int x = %(WS/2)s + x_thread_offset; x < imgWidth - %(WS/2)s; x += %(x_stride)s)
				{
					int i = 0;
					for (int fx = 0; fx < %(WS)s; ++fx)
					{
						for (int fy = 0; fy < %(WS)s; ++fy)
						{
							window[i] = in[(x + fx - %(WS/2)s) + (y + fy - %(WS/2)s)*imgWidth];
							i += 1;
						}
					}

					// Sort to find the median
					for (int j = 0; j < %(WS^2)s; ++j)
					{
						for (int k = j + 1; k < %(WS^2)s; ++k)
						{
							if (window[j] > window[k])
							{
								float tmp = window[j];
								window[j] = window[k];
								window[k] = tmp;
							}
						}
					}
					out[y*imgWidth + x] = window[%(WS^2)s/2];
				}
			}
		}



		__global__ void mf_shared(float* in, float* out, int imgWidth, int imgHeight)
		{
			const int TS = 16 + 2 * %(WS/2)s;
			__shared__ float tile[TS][TS];

			float window[%(WS^2)s];

			const int x_thread_offset = %(BW)s * blockIdx.x + threadIdx.x;
			const int y_thread_offset = %(BH)s * blockIdx.y + threadIdx.y;

			// Load into the tile for this block

			for (int tx = threadIdx.x; tx < TS; tx += 16)
			{
				for (int ty = threadIdx.y; ty < TS; ty += 16)
				{
					if ((x_thread_offset + tx - threadIdx.x) < imgWidth && (y_thread_offset + ty - threadIdx.y) < imgHeight)
					{
						tile[tx][ty] = in[(x_thread_offset + tx - threadIdx.x) + imgWidth * (y_thread_offset + ty - threadIdx.y)];
					}
				}
			}


			for (int y = %(WS/2)s + y_thread_offset; y < imgHeight - %(WS/2)s; y += %(y_stride)s)
			{
				for (int x = %(WS/2)s + x_thread_offset; x < imgWidth - %(WS/2)s; x += %(x_stride)s)
				{
					// Sort to find the median
					for (int j = 0; j < %(WS^2)s; ++j)
					{
						for (int k = j + 1; k < %(WS^2)s; ++k)
						{
							if (window[j] > window[k])
							{
								float tmp = window[j];
								window[j] = window[k];
								window[k] = tmp;
							}
						}
					}
					out[y*imgWidth + x] = window[%(WS^2)s/2];
				}
			}
		}

		"""

	code = code % {
			'BW' : BLOCK_WIDTH,
			'BH' : BLOCK_HEIGHT,
			'WS' : WINDOW_SIZE,
			'WS/2' : WINDOW_SIZE / 2,
			'WS^2' : WINDOW_SIZE * WINDOW_SIZE,
			'x_stride' : BLOCK_WIDTH * gridx,
			'y_stride' : BLOCK_HEIGHT * gridy,
		}

	mod = SourceModule(code)
	mf = mod.get_function('mf_shared')

	indata = np.pad(indata, padding, 'constant', constant_values=0)
	outdata = np.empty_like(indata)

	in_pin = cuda.register_host_memory(indata)


	in_gpu = cuda.mem_alloc(indata.nbytes)
	out_gpu = cuda.mem_alloc(outdata.nbytes)

	cuda.memcpy_htod(in_gpu, in_pin)

	mf.prepare("PPii")
	mf.prepared_call(grid, block, in_gpu, out_gpu, expanded_M, expanded_N)

	cuda.memcpy_dtoh(outdata, out_gpu)


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
				a = indata[i*N:(i+1)*N]
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
				

		outdata = np.concatenate(outdata_list, axis=1)


	if (padding > 0):
		outdata = outdata[padding:-padding, padding:-padding]

	if (timing):
		e.record()
		e.synchronize()
		print "THIS FUNCTION: ", s.time_till(e), "ms"


		s.record()
		true_ans= sps.medfilt2d(indata, (WINDOW_SIZE, WINDOW_SIZE))
		e.record()
		e.synchronize()
		print "SCIPY MEDFILT", s.time_till(e), "ms"

	return outdata

	# return np.allclose(out_pin, true_ans)
	# print out_pin
	# print true_ans