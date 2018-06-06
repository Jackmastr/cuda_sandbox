""" Modified from 
'CUDA median Filtering GPU implementation' by Rajeev Verma
"""

# Eventually get this to work over multiple GPUs, hopefully
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps

BLOCK_WIDTH = 32
BLOCK_HEIGHT = 32

WINDOW_SIZE = 3



code = """
	// X is the 2D image
	const int BW = %(BW)s;
	const int BH = %(BH)s;
	const int WS = %(WS)s;

	__global__ void mf_naive(float *in, float *out, int img_width, int img_height)
	{

	// Set row and column for thread

	int r = blockIdx.y * BH + threadIdx.y;
	int c = blockIdx.x * BW + threadIdx.x;

	int r_stride = BH * gridDim.y;
	int c_stride = BW * gridDim.x;

	float window[WS*WS];   //Take fiter window

	// If the image size is larger than the blocksize, must stride?
	for (int row = r; r < img_height; r += r_stride)
	{
		for (int col = c; c < img_width; c += c_stride)
		{


			if((row==0) || (col==0) || (row==img_height-1) || (col==img_width-1))
						out[row*img_width+col] = 0; //Deal with boundry conditions


			else
			{
				for (int x = 0; x < WS; x++)
				{
					for (int y = 0; y < WS; y++)
					{

						window[x * WS + y] = in[(row + x - 1)*img_width + (col + y - 1)];
					}
				}

				for (int i = 0; i < 9; i++) {
					for (int j = i + 1; j < 9; j++) {
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
}
	"""

code = code % {
		'BW' : BLOCK_WIDTH,
		'BH' : BLOCK_HEIGHT,
		'WS' : WINDOW_SIZE,
	}

mod = SourceModule(code)
mf_naive = mod.get_function('mf_naive')

N = np.int32(29)

#indata = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)
indata = np.array(np.random.rand(N, N), dtype=np.float32)


# Must pad with zeros in order for this strategy to work
indata = np.pad(indata, 1, 'constant', constant_values=0)

outdata = np.empty_like(indata)

in_pin = cuda.register_host_memory(indata)
out_pin = cuda.register_host_memory(outdata)

in_gpu = cuda.mem_alloc(indata.nbytes)
out_gpu = cuda.mem_alloc(outdata.nbytes)

cuda.memcpy_htod(in_gpu, in_pin)
# Check to see if this isn't needed because it is empty
cuda.memcpy_htod(out_gpu, out_pin)

mf_naive.prepare("PPii")

# N + 2 because pad on all sides with zeros
gridx = int(np.ceil((N+2)/BLOCK_WIDTH))
gridy = int(np.ceil((N+2)/BLOCK_HEIGHT))
grid = (1,1)

mf_naive.prepared_call( grid, (BLOCK_WIDTH, BLOCK_HEIGHT, 1), in_gpu, out_gpu, N+2, N+2)


cuda.memcpy_dtoh(out_pin, out_gpu)

true_ans= sps.medfilt2d(indata, (WINDOW_SIZE, WINDOW_SIZE))

out_pin = out_pin[1:-1, 1:-1]
true_ans = true_ans[1:-1, 1:-1]

print np.allclose(out_pin, true_ans)
# print "CUDA OUTPUT"
# print out_pin
# print "SCIPY MEDFILT OUTPUT"
# print true_ans