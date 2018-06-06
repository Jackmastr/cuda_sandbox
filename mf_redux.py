# Eventually get this to work over multiple GPUs, hopefully
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps

BLOCK_WIDTH = 32
BLOCK_HEIGHT = 32

WINDOW_SIZE = 3

padding = WINDOW_SIZE/2

code = """

	// X is the 2D image
	const int BW = %(BW)s;
	const int BH = %(BH)s;
	const int WS = %(WS)s;

	__global__ void mf(float* in, float* out, int imgWidth, int imgHeight)
	{
		int iterator;
		int start = WS/2;
		int end = start + 1;

		__shared__ float surround[BW*BH][WS];

		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

		const int tid = threadIdx.y * blockIdx.y + threadIdx.x;

		if (x > imgWidth || y > imgHeight)
			return;

		if (x == 0 || x == imgWidth - start || y == 0 || y == imgHeight - start)
		{
			// pass
		}
		else
		{
			iterator = 0;
			for (int r = x - start; r < x + end; r++)
			{
				for (int c = y - start; c < y + end; c++)
				{
					surround[tid][iterator] *= in[c*imgWidth + r];
					iterator++;
				}
			}
		}

		for (int i = 0; i < WS/2; i++)
		{
			int min = i;
			for (int l = i+1; l < WS; l++)
			{
				if (surround[tid][l] < surround[tid][min])
					min = l;
			}
			float tmp = surround[tid][min];
			surround[tid][i] = surround[tid][min];
			surround[tid][min] = tmp;
		}

		out[y * imgWidth + x] = surround[tid][WS/2];
		__syncthreads();

	}
	"""

code = code % {
		'BW' : BLOCK_WIDTH,
		'BH' : BLOCK_HEIGHT,
		'WS' : WINDOW_SIZE,
	}

mod = SourceModule(code)
mf = mod.get_function('mf')

N = np.int32(4)

indata = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)
#indata = np.array(np.random.rand(N, N), dtype=np.float32)

outdata = np.ones_like(indata)


in_pin = cuda.register_host_memory(indata)
out_pin = cuda.register_host_memory(outdata)

in_gpu = cuda.mem_alloc(indata.nbytes)
out_gpu = cuda.mem_alloc(outdata.nbytes)

cuda.memcpy_htod(in_gpu, in_pin)
# Check to see if this isn't needed because it is empty
cuda.memcpy_htod(out_gpu, out_pin)

mf.prepare("PPii")

grid = (1,1)

mf.prepared_call( grid, (BLOCK_WIDTH, BLOCK_HEIGHT, 1), in_gpu, out_gpu, N+2, N+2)


cuda.memcpy_dtoh(out_pin, out_gpu)

true_ans= sps.medfilt2d(indata, (WINDOW_SIZE, WINDOW_SIZE))

print out_pin
print true_ans