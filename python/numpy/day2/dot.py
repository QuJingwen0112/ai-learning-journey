import numpy as np
a=np.array([[1,2],[2,1]])
b=np.arange(4).reshape((2,2))
c=a*b
c_dot=np.dot(a,b)
print(c)
print(c_dot)
print(a.shape)
