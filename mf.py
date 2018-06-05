# Eventually get this to work over multiple GPUs, hopefully
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 

BLOCK_WIDTH = 8
BLOCK_HEIGHT = 8

# BW AND BH MUST BOTH BE GREATER THAN WW AND WH

WINDOW_WIDTH = 3
WINDOW_HEIGHT = 3



code = """
	// X is the 2D image
	const int BW = %(BW)s;
	const int BH = %(BH)s;
	const int WW = %(WW)s;
	const int WH = %(WH)s;

	__global__ void medianFilter(float** x, int img_width, int img_height)
	{
		__shared__ float* window[WW * WH];

		int edgex = WW / 2; 
		int edgey = WH / 2;

		const int x = BW * blockIdx.x + threadIdx.x;
		const int y = BH * blockIdx.y + threadIdx.y;

		const int idx = threadIdx.y * BH + threadIdx.x;

		// Now we fill up the window ASSUMING BW >= WW 
		window

		__syncthreads();

	}
	"""