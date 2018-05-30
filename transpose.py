import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

TILE_DIM = 32
BLOCK_ROWS = 8

N = 32 * 128

idata = np.tril(np.ones((N,N), dtype=np.float32))
idata = idata.flatten()

odata = np.empty(N*N, dtype=np.float32)

idata_pin = cuda.register_host_memory(idata)
odata_pin = cuda.register_host_memory(odata)

idata_gpu = cuda.mem_alloc(idata.nbytes)
odata_gpu = cuda.mem_alloc(odata.nbytes)

cuda.memcpy_htod_async(idata_gpu, idata_pin)
cuda.memcpy_htod_async(odata_gpu, odata_pin)

code = """
__global__ void copy(float *odata, const float *idata)
{	
  const int TILE_DIM = %d;
  const int BLOCK_ROWS = %d;

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y *  TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < 32; j+= BLOCK_ROWS)
    odata[(y+j)*width + x] = idata[(y+j)*width + x];
}
	"""
code = code % (TILE_DIM, BLOCK_ROWS)
mod = SourceModule(code)

copy = mod.get_function("copy")

grid = (N/TILE_DIM, N/TILE_DIM, 1)
block = (TILE_DIM, BLOCK_ROWS, 1)

copy.prepare("PP")
copy.prepared_call(grid, block, odata_gpu, idata_gpu)

cuda.memcpy_dtoh(odata_pin, odata_gpu)

odata_pin = np.reshape(odata_pin, (N,N))

print odata_pin
