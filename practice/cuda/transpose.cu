#include <stdio.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

__global__ void copy(float *odata, const float *idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;
	
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		// row is (tile y + offset into tile j)*width of matrix + col x
		odata[(y+j)*width + x] = idata[(y+j)*width + x];
	}
}

__global__ void transposeNaive(float *odata, const float *idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		odata[x*width + (y+j)] = idata[(y+j)*width + x];
	}
}

__global__ void transposeCoalesced(float *odata, float *idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
	}

	__syncthreads();
	
	// Now these are the offsets into the transposed tile
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
	}
}
