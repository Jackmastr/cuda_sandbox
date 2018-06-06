# Eventually get this to work over multiple GPUs, hopefully
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps

BLOCK_WIDTH = 8
BLOCK_HEIGHT = 8

WINDOW_SIZE = 1



code = """
	// X is the 2D image
	const int BW = %(BW)s;
	const int BH = %(BH)s;
	const int WS = %(WS)s;

	__global__ void mf_pad(float *in, float *out, int img_width, int img_height)
	{
		int nx, ny, hN[2];
		int pre_x, pre_y, pos_x, pos_y;
		int subx, suby, k, totN;
		float *fptr1, *fptr2, *ptr1, *ptr2;

		totN = img_width * img_height;
		float myvals[totN];

		for (int y = 0; y < img_height; y++)
		{
			for (int x = 0; x < img_width; x++)
			{
				
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
mf_naive = mod.get_function('mf_pad')

N = np.int32(4)

indata = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)

outdata = np.empty_like(indata)


in_pin = cuda.register_host_memory(indata)
out_pin = cuda.register_host_memory(outdata)


in_gpu = cuda.mem_alloc(indata.nbytes)
out_gpu = cuda.mem_alloc(outdata.nbytes)


cuda.memcpy_htod(in_gpu, in_pin)
# Check to see if this isn't needed because it is empty
cuda.memcpy_htod(out_gpu, out_pin)


mf_naive.prepare("PPii")
mf_naive.prepared_call((1,1,1), (BLOCK_WIDTH, BLOCK_HEIGHT, 1), in_gpu, out_gpu, N, N)


cuda.memcpy_dtoh(out_pin, out_gpu)


true_ans= sps.medfilt2d(indata, (WINDOW_SIZE, WINDOW_SIZE))
