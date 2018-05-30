import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

s = cuda.Event()
e = cuda.Event()

TILE_DIM = 32
BLOCK_ROWS = 8

N = 32 * 1024

idata = np.tril(np.ones((N,N), dtype=np.float32))

odata = np.empty_like(idata, dtype=np.float32)

idata_pin = cuda.register_host_memory(idata)
odata_pin = cuda.register_host_memory(odata)

idata_gpu = cuda.mem_alloc(idata.nbytes)
odata_gpu = cuda.mem_alloc(odata.nbytes)

cuda.memcpy_htod_async(idata_gpu, idata_pin)
cuda.memcpy_htod_async(odata_gpu, odata_pin)


code = """
const int TILE_DIM = %d;
const int BLOCK_ROWS = %d;

__global__ void copy(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y *  TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[(y+j)*width + x] = idata[(y+j)*width + x];
}

__global__ void copySharedMem(float *odata, const float *idata)
	{
		__shared__ float tile[TILE_DIM * TILE_DIM];
		int x = blockIdx.x * TILE_DIM + threadIdx.x;
		int y = blockIdx.y *  TILE_DIM + threadIdx.y;
		int width = gridDim.x * TILE_DIM;

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
			tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];

		__syncthreads();

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
			odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];

	}

__global__ void transpose(float *odata, const float *idata)
	{
		__shared__ float tile[TILE_DIM][TILE_DIM+1];
		int x = blockIdx.x * TILE_DIM + threadIdx.x;
		int y = blockIdx.y *  TILE_DIM + threadIdx.y;
		int width = gridDim.x * TILE_DIM;

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
			tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

		__syncthreads();

		// Now have to index into the tile matrix as well
		x = blockIdx.y * TILE_DIM + threadIdx.x;
		y = blockIdx.x * TILE_DIM + threadIdx.y;

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
			odata[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
	}
	"""
code = code % (TILE_DIM, BLOCK_ROWS)
mod = SourceModule(code)

copy = mod.get_function("copy")

grid = (N/TILE_DIM, N/TILE_DIM, 1)
block = (TILE_DIM, BLOCK_ROWS, 1)
nStreams = 2


stream = []
for i in range(nStreams):
	stream.append(cuda.Stream())


copy.prepare("PP")
copy.prepared_call(grid, block, odata_gpu, idata_gpu)

cuda.memcpy_dtoh(odata_pin, odata_gpu)

print odata_pin