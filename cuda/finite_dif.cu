#include <stdio.h>

int mx, my, mz = 64;
int sPencils = 4; // but this is a parameter you can tune


// for these constants need to load them with a cuda command l8r
// in pyCUDA use memcpy_htod(...)?
__constant__ float c_ax, c_bx, c_cx, c_dx;
__constant__ float c_ay, c_by, c_cy, c_dy;
__constant__ float c_az, c_bz, c_cz, c_dz;

// stencil weights for the coefficients
float dsinv = mx - 1.f; // Delta x

float ax =  4.f / 5.f   * dsinv;
float bx = -1.f / 5.f   * dsinv;
float cx =  4.f / 105.f * dsinv;
float dx = -1.f / 280.f * dsinv;

cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyHostToDevice);
cudaMemcpyToSymbol(c_bx, &bx, sizeof(float), 0, cudaMemcpyHostToDevice);
cudaMemcpyToSymbol(c_cx, &cx, sizeof(float), 0, cudaMemcpyHostToDevice);
cudaMemcpyToSymbol(c_dx, &dx, sizeof(float), 0, cudaMemcpyHostToDevice);

__global__ void derivative_x(float *f, float *df)
{
	__shared__ float s_f[sPencils][mx + 8]; // 4-wide halo

	int i = threadIdx.x; // row index for matrix
	int j = blockIdx.x * blockDim.y + threadIdx.y; // col index
	int k = blockIdx.y; // depth index

	int si = i + 4; // local i for shared mem
	int sj = threadIdx.y; // local j for shared mem

	// mx, my, mz are the array dimensions 
	int globalIdx = (k * mx * my) + (j * mx) + i;

	s_f[sj][si] = f[globalIdx];

	__syncthreads();

	// fills in the ?periodic? images in shared memory array
	// cause we are using periodic boundary conditions for the derivative
	if (i < 4)
	{
		s_f[sj][si - 4] = s_f[sj][si + mx - 5];
		s_f[sj][si + mx] = s_f[sj][si+1];
	}

	__syncthreads();

	// This is just the finite dif equation we are using
	// for the x-direction
	df[globalIdx] = 
		( c_ax * ( s_f[sj][si + 1] - s_f[sj][si-1] ) )
		( c_bx * ( s_f[sj][si + 2] - s_f[sj][si-2] ) )
		( c_cx * ( s_f[sj][si + 3] - s_f[sj][si-3] ) )
		( c_dx * ( s_f[sj][si + 4] - s_f[sj][si-4] ) );
}