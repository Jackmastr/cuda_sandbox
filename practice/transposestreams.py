import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

s = cuda.Event()
e = cuda.Event()

TILE_DIM = 32
BLOCK_ROWS = 8

N = 32 * 256



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
transpose = mod.get_function("transpose")

nStreams = 8

grid = (N/(nStreams*TILE_DIM), N/(TILE_DIM), 1)
#grid = (1, 1, 1)
block = (TILE_DIM, BLOCK_ROWS, 1)


stream = []
for i in range(nStreams):
	stream.append(cuda.Stream())

idata = np.tril(np.ones((N,N), dtype=np.float32))

odata = np.zeros_like(idata, dtype=np.float32)

idata_pin_list = []
odata_pin_list = []

for i in range(nStreams):
	i_slice = idata[i*N/nStreams:(i+1)*N/nStreams]
	o_slice = odata[i*N/nStreams:(i+1)*N/nStreams]
	idata_pin_list.append(cuda.register_host_memory(i_slice))
	odata_pin_list.append(cuda.register_host_memory(o_slice))

#print idata_pin_list[0]

# Using independent host->device, kernel, device->host streams?
idata_gpu_list = []
odata_gpu_list = []

for i in range(nStreams):
	idata_gpu_list.append(cuda.mem_alloc(idata.nbytes/nStreams))
	odata_gpu_list.append(cuda.mem_alloc(odata.nbytes/nStreams))

	cuda.memcpy_htod_async(idata_gpu_list[i], idata_pin_list[i])
	cuda.memcpy_htod_async(odata_gpu_list[i], odata_pin_list[i])

	stream[i].synchronize()

	copy.prepare("PP")
	copy.prepared_call(grid, block, odata_gpu_list[i], idata_gpu_list[i])

	stream[i].synchronize()

	cuda.memcpy_dtoh_async(odata_pin_list[i], odata_gpu_list[i])

for i in range(nStreams):
	stream[i].synchronize()

odata_pin_list = np.asarray(odata_pin_list)
odata_pin_list = odata_pin_list.reshape((N,N))

print odata_pin_list