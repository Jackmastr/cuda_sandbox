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


	code = """
		// X is the 2D image
		const int BW = %(BW)s;
		const int BH = %(BH)s;
		const int WS = %(WS)s;

		// From redzhepdx's quick select implementation
		#define ELEM_SWAP(a,b) {register float t=(a);(a)=(b);(b)=t;}

		__device__ float quickSelectMedian(float *arr, int size)
		{
			int low, high;
			int median;
			int middle, ll, hh;

			float value;

			low = 0;
			high = size - 1;
			median = (low + high) / 2;

			for (int i = 0; i < WS*WS; ++i)
			{
				if (high <= low) // For 1 elem arrays
				{
					value = arr[median];
					return value;
				}

				if (high == low + 1) // For 2 elem arrays
				{
					if (arr[low] > arr[high])
					{
						ELEM_SWAP(arr[low], arr[high]);
					}
					value = arr[median];
					return value;
				}

				middle = (low + high) / 2;

				if (arr[middle] > arr[high]) ELEM_SWAP(arr[middle], arr[high]);
				if (arr[low] > arr[high]) ELEM_SWAP(arr[low], arr[high]);
				if (arr[middle] > arr[low]) ELEM_SWAP(arr[middle], arr[low]);

				ELEM_SWAP(arr[middle], arr[low + 1]);

				ll = low + 1;
				hh = high;

				while (1)
				{
					while (arr[low] > arr[ll])
						ll++;
					while (arr[hh] > arr[low])
						hh--;

					if (hh < ll) break;

					ELEM_SWAP(arr[ll], arr[hh]);
				}

				ELEM_SWAP(arr[low], arr[hh]);

				if (hh <= median) low = ll;
				if (hh >= median) high = hh - 1;
				break;
			}
			return 0;
		}


		__global__ void mf(float* in, float* out, int imgWidth, int imgHeight)
		{
			const int med = (WS*WS)/2;

			float window[WS*WS];

			const int edgex = WS/2; // window width / 2
			const int edgey = WS/2; // window height / 2

			const int x_thread_offset = BW * blockIdx.x + threadIdx.x;
			const int y_thread_offset = BH * blockIdx.y + threadIdx.y;

			const int x_stride = BW * gridDim.x;
			const int y_stride = BH * gridDim.y;


			for (int y = edgey + y_thread_offset; y < imgHeight - edgey; y += y_stride)
			{
				for (int x = edgex + x_thread_offset; x < imgWidth - edgex; x += x_stride)
				{
					int i = 0;
					for (int fx = 0; fx < WS; ++fx)
					{
						for (int fy = 0; fy < WS; ++fy)
						{
							window[i] = in[(x + fx - edgex) + (y + fy - edgey)*imgWidth];
							i += 1;
						}
					}

					// Sort to find the median
					for (int j = 0; j < WS*WS; ++j)
					{
						for (int k = j + 1; k < WS*WS; ++k)
						{
							if (window[j] > window[k])
							{
								float tmp = window[j];
								window[j] = window[k];
								window[k] = tmp;
							}
						}
					}
					out[y*imgWidth + x] = window[med];
				}
			}
		}
		"""

	code = code % {
			'BW' : BLOCK_WIDTH,
			'BH' : BLOCK_HEIGHT,
			'WS' : WINDOW_SIZE,
		}

	mod = SourceModule(code)
	mf = mod.get_function('mf')

	indata = np.pad(indata, padding, 'constant', constant_values=0)
	outdata = np.ones_like(indata)

	in_pin = cuda.register_host_memory(indata)
	out_pin = cuda.register_host_memory(outdata)

	if (nStreams > 0):
		N = N/nStreams
		N_lo = N % nStreams# leftover if N doesn't divide evenly into the streams

		stream = []
		indata_list = []
		outdata_list = []
		for i in xrange(nStreams):
			stream.append(cuda.Stream())

			indata_list.append(indata[i*N:(i+1)*N])
			outdata_list.append(outdata[i*N:(i+1)*N])

			if (i == nStreams - 1):
				indata_list[i] = np.concatenate(indata_list[i], indata[(i+1)*N:])
				outdata_list[i] = np.concatenate(outdata_list[i], outdata[(i+1)*N:])






	in_gpu = cuda.mem_alloc(indata.nbytes)
	out_gpu = cuda.mem_alloc(outdata.nbytes)

	cuda.memcpy_htod(in_gpu, in_pin)
	# Check to see if this isn't needed because it is empty
	cuda.memcpy_htod(out_gpu, out_pin)

	mf.prepare("PPii")

	expanded_N = N + (2 * padding)
	expanded_M = M + (2 * padding)

	gridx = max(1, int(np.ceil((expanded_N)/BLOCK_WIDTH)))
	gridy = max(1, int(np.ceil((expanded_M)/BLOCK_HEIGHT)))
	grid = (gridx,gridy)
	block = (BLOCK_WIDTH, BLOCK_HEIGHT, 1)
	
	mf.prepared_call(grid, block, in_gpu, out_gpu, expanded_M, expanded_N)


	cuda.memcpy_dtoh(out_pin, out_gpu)

	if (padding > 0):
		out_pin = out_pin[padding:-padding, padding:-padding]

	if (timing):
		e.record()
		e.synchronize()
		print "THIS FUNCTION: ", s.time_till(e), "ms"


		s.record()
		true_ans= sps.medfilt2d(indata, (WINDOW_SIZE, WINDOW_SIZE))
		e.record()
		e.synchronize()
		print "SCIPY MEDFILT", s.time_till(e), "ms"

	return out_pin

	# return np.allclose(out_pin, true_ans)
	# print out_pin
	# print true_ans