import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np 
import pycuda.autoinit

code = """
	#include <cuComplex.h>
	__global__ void MatMulKernel(cuFloatComplex *A, cuFloatComplex *B, cuFloatComplex *C)
	{
		const int widthA = %(MATRIX_SIZE)s;
		const int widthB = %(MATRIX_SIZE)s;

		const int blockx = blockIdx.x;
		const int blocky = blockIdx.y;

		const int threadx = threadIdx.x;
		const int thready = threadIdx.y;

		// Sub-matrix of A start
		const int aStart = widthA * %(BLOCK_SIZE)s * blocky;
		// Sub-matrix of A end
		const int aEnd = aStart + widthA - 1;
		
	}
	"""