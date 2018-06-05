import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

start = cuda.Event()
end = cuda.Event()


n = np.int32(1e6)
a = np.float32(2)

start.record()
x_gpu = gpuarray.to_gpu(np.ones(n, dtype=np.float32))
y_gpu = gpuarray.to_gpu(2.*np.ones(n, dtype=np.float32))
end.record()
end.synchronize()
print "Transfer to device takes: ", start.time_till(end), " ms"

start.record()
saxpy_gpu = a*x_gpu + y_gpu
end.record()
end.synchronize()
print "Calculation on device takes: ", start.time_till(end), " ms"

start.record()
saxpy = saxpy_gpu.get()
end.record()
end.synchronize()
print "Transfer to host takes: ", start.time_till(end), " ms"


print np.max(np.abs(4-saxpy))
