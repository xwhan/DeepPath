import numpy as np 

M = np.loadtxt('A.bern')
M = M.reshape([-1,100,100])
print M[0,:,:].shape