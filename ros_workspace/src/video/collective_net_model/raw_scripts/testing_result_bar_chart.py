import numpy as np
import matplotlib.pyplot as plt

num = 1
x = np.load("range2/{0}_in_range.npy".format(num))
y = np.load("range2/{0}_out_range.npy".format(num))

z = np.load("range2/{0}_median.npy".format(num))
print x.shape
print y.shape
avg_1 = np.sum(x[:,0:num+1],axis=1)
avg_2 = np.sum(y[:,0:num+1],axis=1)
print np.mean(avg_1)
print np.mean(avg_2)
print z

plt.hold(True)

plt.plot(avg_1, color="blue")
plt.plot(avg_2, color="green")

# plt.hist(avg_1, color="blue")
# plt.hist(avg_2, color="green")

plt.show()

