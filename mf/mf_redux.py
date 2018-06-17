#!/usr/bin/env python

# Eventually get this to work over multiple GPUs, hopefully

# Test without using so many format stream tricks, it may be making it slower

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps


def MedianFilter(input=None, kernel_size=3, bw=32, bh=32, n=256, m=0, timing=False, nStreams=0, input_list=None):

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

	gridx = int(np.ceil((expanded_N)/BLOCK_WIDTH))+1
	gridy = int(np.ceil((expanded_M)/BLOCK_HEIGHT))+1
	grid = (gridx,gridy, 1)
	block = (BLOCK_WIDTH, BLOCK_HEIGHT, 1)

	code = """
		#include <stdio.h>
		#pragma comment(linker, "/HEAP:4000000")

		__device__ float FloydWirth_kth(float arr[], const int length, const int kTHvalue) 
		{
		#define F_SWAP(a,b) { float temp=(a);(a)=(b);(b)=temp; }
		#define SIGNUM(x) ((x) < 0 ? -1 : ((x) > 0 ? 1 : (x)))

		    int left = 0;       
		    int right = length - 1;     
		    int left2 = 0;
		    int right2 = length - 1;

		    while (left < right) 
		    {           
		        if( arr[right2] < arr[left2] ) F_SWAP(arr[left2],arr[right2]);
		        if( arr[right2] < arr[kTHvalue] ) F_SWAP(arr[kTHvalue],arr[right2]);
		        if( arr[kTHvalue] < arr[left2] ) F_SWAP(arr[left2],arr[kTHvalue]);

		        int rightleft = right - left;

		        if (rightleft < kTHvalue)
		        {
		            int n = right - left + 1;
		            int ii = kTHvalue - left + 1;
		            int s = (n + n) / 3;
		            int sd = (n * s * (n - s) / n) * SIGNUM(ii - n / 2);
		            int left2 = max(left, kTHvalue - ii * s / n + sd);
		            int right2 = min(right, kTHvalue + (n - ii) * s / n + sd);              
		        }

		        float x=arr[kTHvalue];

		        while ((right2 > kTHvalue) && (left2 < kTHvalue))
		        {
		            do 
		            {
		                left2++;
		            }while (arr[left2] < x);

		            do
		            {
		                right2--;
		            }while (arr[right2] > x);

		            F_SWAP(arr[left2],arr[right2]);
		        }
		        left2++;
		        right2--;

		        if (right2 < kTHvalue) 
		        {
		            while (arr[left2]<x)
		            {
		                left2++;
		            }
		            left = left2;
		            right2 = right;
		        }

		        if (kTHvalue < left2) 
		        {
		            while (x < arr[right2])
		            {
		                right2--;
		            }

		            right = right2;
		            left2 = left;
		        }

		        if( arr[left] < arr[right] ) F_SWAP(arr[right],arr[left]);
		    }

		#undef F_SWAP
		#undef SIGNUM
		    return arr[kTHvalue];
		}




		__global__ void mf(float* in, float* out, int imgDimY, int imgDimX)
		{

			float window[%(WS^2)s];

			const int x_thread_offset = %(BY)s * blockIdx.x + threadIdx.x;
			const int y_thread_offset = %(BX)s * blockIdx.y + threadIdx.y;

			for (int y = %(WSx/2)s + y_thread_offset; y < imgDimX - %(WSx/2)s; y += %(y_stride)s)
			{
				for (int x = %(WSy/2)s + x_thread_offset; x < imgDimY - %(WSy/2)s; x += %(x_stride)s)
				{
					int i = 0;
					for (int fx = 0; fx < %(WSy)s; ++fx)
					{
						for (int fy = 0; fy < %(WSx)s; ++fy)
						{
							window[i] = in[(x + fx - %(WSy/2)s) + (y + fy - %(WSx/2)s)*imgDimY];
							i += 1;
						}
					}

					// Sort to find the median
					//for (int j = 0; j < %(WS^2)s; ++j)
					//{
					//	for (int k = j + 1; k < %(WS^2)s; ++k)
					//	{
					//		if (window[j] > window[k])
					//		{
					//			float tmp = window[j];
					//			window[j] = window[k];
					//			window[k] = tmp;
					//		}
					//	}
					//}
					//out[y*imgDimY + x] = window[%(WS^2)s/2];
					out[y*imgDimY + x] = FloydWirth_kth(window, %(WS^2)s, %(WS^2)s/2);
				}
			}
		}

		__global__ void mf_shared(float* in, float* out, int imgDimY, int imgDimX)
		{
			const int TSx = %(BX)s + %(WSx)s - 1;
			const int TSy = %(BY)s + %(WSy)s - 1;
            __shared__ float tile[TSx][TSy];

            float window[%(WS^2)s];
            const int x_thread_offset = %(BX)s * blockIdx.x + threadIdx.x;
            const int y_thread_offset = %(BY)s * blockIdx.y + threadIdx.y;


			const int thread_index = blockDim.y * threadIdx.x + threadIdx.y;

			int imgX = blockIdx.x * blockDim.x + thread_index;
			int imgY;


            // Load into the tile for this block
			if (thread_index < TSx && imgX < imgDimX)
			{
				for (int i = 0; i < TSy && i < imgDimY - blockIdx.y * blockDim.y; i++)
				{
					imgY = blockIdx.y * blockDim.y + i;
					tile[thread_index][i] = in[imgX * imgDimY + imgY];
				}

			}

			__syncthreads();


			int x = %(WSx/2)s + x_thread_offset;
			int y = %(WSy/2)s + y_thread_offset;

			if (x >= imgDimX - %(WSx/2)s || y >= imgDimY - %(WSy/2)s)
			{
				return;
			}

			int i = 0;
			for (int fx = 0; fx < %(WSy)s; ++fx)
			{
				for (int fy = 0; fy < %(WSx)s; ++fy)
				{
					window[i++] = tile[threadIdx.x + fx][threadIdx.y + fy];
				}
			}


			// Sort to find the median
			//for (int j = 0; j <= %(WS^2)s/2; j++)
			//{
			//	for (int k = j + 1; k < %(WS^2)s; k++)
			//	{
			//		if (window[j] > window[k])
			//		{
			//			float tmp = window[j];
			//			window[j] = window[k];
			//			window[k] = tmp;
			//		}
			//	}
			//}
			//out[x*imgDimY + y] = window[%(WS^2)s/2];

			out[x*imgDimY + y] = FloydWirth_kth(window, %(WS^2)s, %(WS^2)s/2);
		}

		/* MEDIAN FILTER METHODS FROM https://github.com/aglenis/gpu_medfilter/blob/master/cuda_implementation/median_shared.cu */

		__device__ void insertionSort(float window[],int size)
		{
    		int i , j;
    		float temp;
    		for(i = 0; i < size; i++) {
        		temp = window[i];
        		for(j = i-1; j >= 0 && temp < window[j]; j--) {
            		window[j+1] = window[j];
        		}
        		window[j+1] = temp;
    		}
		}

		__global__ void MedianFilter2D( float *input,float* output,int widthImage, int heightImage)
		{
    		int filter_offset=%(WS^2)s/2;
			//y and x are oposite the cuda programming model
    		unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    		unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    		if(y>heightImage || x>widthImage)
        		return;

    		float window[%(WS^2)s];
   			for (int counter=0; counter<%(WS^2)s; counter++)
    		{
        		window[counter]=0;
    		}
    		int count=0;
    		for( int k=max(y-filter_offset,0); k<=min(y+filter_offset,heightImage-1); k++)
    		{
        		for (int l=max(x-filter_offset,0); l<=min(x+filter_offset,widthImage-1); l++)
        		{

            		window[count++]=input[(k)*widthImage+(l)];

        		}
    		}
    		insertionSort(window, %(WS^2)s);

    		output[y*widthImage + x]=window[%(WS^2)s/2];

}


		"""

	code = code % {
			'BY' : BLOCK_WIDTH,
			'BX' : BLOCK_HEIGHT,
			'WS^2' : WS_x * WS_y,
			'x_stride' : BLOCK_WIDTH * gridx,
			'y_stride' : BLOCK_HEIGHT * gridy,
			'WSx' : WS_x,
			'WSy' : WS_y,
			'WSx/2' : WS_x/2,
			'WSy/2' : WS_y/2
		}
	# s.record()
	mod = SourceModule(code)
	#mf = mod.get_function('mf')
	mf_shared = mod.get_function('mf')
	# e.record()
	# e.synchronize()
	# print s.time_till(e), "ms"


	# NSTREAMS := NUMBER OF INPUT IMAGES
	if (nStreams > 0):




		# Initialize the streams
		stream = [cuda.Stream()]*nStreams

		# Pad all the images with zeros
		input_list = [np.array( np.pad(img, ( (padding_y, padding_y), (padding_x, padding_x) ), 'constant', constant_values=0) , dtype=np.float32) for img in input_list]

		# Use pinned memory for all the images
		in_pin_list = [cuda.register_host_memory(img) for img in input_list]
		imgBytes = in_pin_list[0].nbytes

		# Initialize the outputs to empty images (assuming all images are of the same shape)
		outdata_list = [cuda.pagelocked_empty_like(img) for img in input_list]

		# Malloc on the GPU for each input and output image
		#in_gpu_list = [cuda.mem_alloc(pinnedImg.nbytes) for pinnedImg in in_pin_list]
		in_gpu_list = [None]*nStreams
		#out_gpu_list = [cuda.mem_alloc(pinnedImg.nbytes) for pinnedImg in in_pin_list]
		out_gpu_list = [None]*nStreams

		for i in xrange(nStreams + 2):
			ii = i - 1
			iii = i - 2

			if 0 <= iii < nStreams:
				st = stream[iii]
				# s.record(stream=stream[5])
				cuda.memcpy_dtoh_async(outdata_list[iii], out_gpu_list[iii], stream=st)

			if 0 <= ii < nStreams:
				st = stream[ii]
				out_gpu_list[ii] = cuda.mem_alloc(imgBytes)
				# s.record(stream=stream[5])
				mf_shared.prepare("PPii")
				mf_shared.prepared_async_call(grid, block, st, in_gpu_list[ii], out_gpu_list[ii], expanded_M, expanded_N)
				# e.record(stream=stream[5])
				# e.synchronize()
				#print s.time_till(e), "ms for the kernel"

			if 0 <= i < nStreams:
				st = stream[i]
				# s.record(stream=stream[5])
				in_gpu_list[i] = cuda.mem_alloc(imgBytes)
				cuda.memcpy_htod_async(in_gpu_list[i], in_pin_list[i], stream=st)
				# e.record(stream=stream[5])
				# e.synchronize()
				# print s.time_till(e), "ms for the transfer"

		if (padding_y > 0):
			outdata_list = [out[padding_y:-padding_y] for out in outdata_list]
		if (padding_x > 0):
			outdata_list = [out[:, padding_x:-padding_x] for out in outdata_list]

		return outdata_list





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

	ks = cuda.Event()
	ke = cuda.Event()

	#mf.prepare("PPii")
	mf_shared.prepare("PPii")

	# ks.record()
	#mf.prepared_call(grid, block, in_gpu, out_gpu, expanded_M, expanded_N)
	# ke.record()
	# ke.synchronize()
	# print "UNSHARED:", ks.time_till(ke), "ms"

	# ks.record()
	mf_shared.prepared_call(grid, block, in_gpu, out_gpu, expanded_M, expanded_N)
	# ke.record()
	# ke.synchronize()
	# print "SHARED: ", ks.time_till(ke), "ms"




	cuda.memcpy_dtoh(out_pin, out_gpu)

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
