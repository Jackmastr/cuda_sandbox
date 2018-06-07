# Eventually get this to work over multiple GPUs, hopefully
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np 
import scipy.signal as sps

BLOCK_WIDTH = 32
BLOCK_HEIGHT = 32

WINDOW_SIZE = 5

padding = WINDOW_SIZE/2

N = np.int32(256)

#indata = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)
indata_orig = np.array(np.random.rand(N, N), dtype=np.float32)

s = cuda.Event()
e = cuda.Event()
s.record()


code = """

	// X is the 2D image
	const int BW = %(BW)s;
	const int BH = %(BH)s;
	const int WS = %(WS)s;

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


		for (int x = edgex + x_thread_offset; x < imgWidth - edgex; x += x_stride)
		{
			for (int y = edgey + y_thread_offset; y < imgWidth - edgey; y += y_stride)
			{
				int i = 0;
				for (int fx = 0; fx < WS; ++fx)
				{
					for (int fy = 0; fy < WS; ++fy)
					{
						window[i] = in[(x + fx - edgex)*imgWidth + (y + fy - edgey)];
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

				out[x*imgWidth + y] = window[med];

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




indata = np.pad(indata_orig, padding, 'constant', constant_values=0)
outdata = np.ones_like(indata)


in_pin = cuda.register_host_memory(indata)
out_pin = cuda.register_host_memory(outdata)

in_gpu = cuda.mem_alloc(indata.nbytes)
out_gpu = cuda.mem_alloc(outdata.nbytes)

cuda.memcpy_htod(in_gpu, in_pin)
# Check to see if this isn't needed because it is empty
cuda.memcpy_htod(out_gpu, out_pin)

mf.prepare("PPii")

gridx = int(np.ceil((N+2*padding)/BLOCK_WIDTH))
gridy = int(np.ceil((N+2*padding)/BLOCK_HEIGHT))
grid = (gridx,gridy)

mf.prepared_call( grid, (BLOCK_WIDTH, BLOCK_HEIGHT, 1), in_gpu, out_gpu, N+2*padding, N+2*padding)


cuda.memcpy_dtoh(out_pin, out_gpu)

if (padding > 0):
	out_pin = out_pin[padding:-padding, padding:-padding]


e.record()
e.synchronize()
print s.time_till(e), "ms"


s.record()
true_ans= sps.medfilt2d(indata_orig, (WINDOW_SIZE, WINDOW_SIZE))
e.record()
e.synchronize()
print s.time_till(e), "ms"

print np.allclose(out_pin, true_ans)
print out_pin
print true_ans