""" Modified from 
'CUDA median Filtering GPU implementation' by Rajeev Verma
"""

# Eventually get this to work over multiple GPUs, hopefully
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps

s = cuda.Event()
e = cuda.Event()

BLOCK_WIDTH = 1
BLOCK_HEIGHT = 1

WINDOW_SIZE = 3

padding = WINDOW_SIZE/2

code = """

	#include <stdio.h>

	// X is the 2D image
	//const int BW = %(BW)s;
	//const int BH = %(BH)s;
	const int WS = %(WS)s;

	__global__ void mf_naive(float *in, float *out, int img_width, int img_height)
	{

	// Set row and column for thread

	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	int r_stride = blockDim.y * gridDim.y;
	int c_stride = blockDim.x * gridDim.x;

	float window[WS*WS];   //Take fiter window

	// If the image size is larger than the blocksize, must stride?
	for (int row = r; row < img_height; row += r_stride)
	{
		for (int col = c; col < img_width; col += c_stride)
		{
			for (int x = 0; x < WS; x++)
			{
				for (int y = 0; y < WS; y++)
				{

					window[x * WS + y] = in[(row + x)*img_width + (col + y)];
				}
			}

			for (int i = 0; i < WS*WS; i++) {
				for (int j = i + 1; j < WS*WS; j++) {
					if (window[i] > window[j]) { 
						//Swap the variables.
						float tmp = window[i];
						window[i] = window[j];
						window[j] = tmp;
					}
				}
			}

			out[row*img_width+col] = window[(WS*WS)/2];   //Set the output variables.

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
mf_naive = mod.get_function('mf_naive')

N = np.int32(1)

#indata = np.array([[2]], dtype=np.float32)
indata = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)
#indata = np.array(np.random.rand(N, N), dtype=np.float32)

s.record()
# Must pad with zeros in order for this strategy to work
indata = np.pad(indata, padding, 'constant', constant_values=0)
outdata = np.ones_like(indata)


in_pin = cuda.register_host_memory(indata)
out_pin = cuda.register_host_memory(outdata)

in_gpu = cuda.mem_alloc(indata.nbytes)
out_gpu = cuda.mem_alloc(outdata.nbytes)

cuda.memcpy_htod(in_gpu, in_pin)
# Check to see if this isn't needed because it is empty
cuda.memcpy_htod(out_gpu, out_pin)

mf_naive.prepare("PPii")

# N + 2 because pad on all sides with zeros
gridx = int(np.ceil((N+2*padding)/BLOCK_WIDTH))
gridy = int(np.ceil((N+2*padding)/BLOCK_HEIGHT))
grid = (gridx,gridy)
grid = (1,1)



mf_naive.prepared_call( grid, (BLOCK_WIDTH, BLOCK_HEIGHT, 1), in_gpu, out_gpu, N+2*padding, N+2*padding)


cuda.memcpy_dtoh(out_pin, out_gpu)

e.record()
e.synchronize()
#print s.time_till(e), "ms"


s.record()
true_ans= sps.medfilt2d(indata, (WINDOW_SIZE, WINDOW_SIZE))
e.record()
e.synchronize()
#print s.time_till(e), "ms"


if (padding > 0):
	out_pin = out_pin[padding:-padding, padding:-padding]
	true_ans = true_ans[padding:-padding, padding:-padding]

# print np.allclose(out_pin, true_ans)
# print "CUDA OUTPUT"
# print out_pin
# print "SCIPY MEDFILT OUTPUT"
# print true_ans