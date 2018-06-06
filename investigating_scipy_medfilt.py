import numpy as np 
import scipy.signal as sps

a = np.array([[2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3], [2, 80, 6, 3]], dtype=np.float32)
b = sps.medfilt2d(a, 3)
print b