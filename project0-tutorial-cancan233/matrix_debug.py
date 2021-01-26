import numpy as np
from numpy import random as r

mat_1 = r.rand(200,150)
mat_2 = mat_1
mat_2 = np.transpose(mat_2)
mat_3 = np.matmul(mat_1, mat_2)
mat_max = np.max(mat_3)

print("Max value:", mat_max)