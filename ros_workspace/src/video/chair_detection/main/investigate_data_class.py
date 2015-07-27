import numpy as np 
# x = np.load("data_x3.npy")
y = np.load("data_y3.npy")

# z= np.array([1,2,3,4])
print y.shape

x = np.load("data_x3.npy")
print x.shape
z = np.bincount(y)
print z
