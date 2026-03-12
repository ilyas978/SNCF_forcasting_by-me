import numpy as np 
L=np.array([1, 0])
a=np.where(L==0, -1, 1 )
print(a*L)