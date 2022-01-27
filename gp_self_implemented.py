import numpy as np
from matplotlib import pyplot as plt
kernel_types = ["linear", "brownian", "sq_exp", "orn-uhl", "periodic", "symmetric"]
k_num = 5

def kernel(x,y, type = "linear"):
    if type == kernel_types[0]:
        return np.dot(x.T, y)
    elif type == kernel_types[1]:
        return min(x, y)
    elif type == kernel_types[2]:
        alpha = 100
        return np.exp(-alpha*np.dot((x-y).T,(x-y)))
    elif type == kernel_types[3]:
        alpha = 1
        return np.exp(-alpha*np.sqrt(np.dot((x-y).T, (x-y))))
    elif type == kernel_types[4]:
        alpha = 1
        beta = 5
        return np.exp(-alpha*np.sin(beta*np.pi*(x-y))**2)
    elif type == kernel_types[5]:
        alpha = 100
        return np.exp(-alpha*min(abs(x-y), abs(x+y))**2)

x = np.linspace(-1, 1, 200)
n = len(x)

C = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        C[i,j] = kernel(x[i], x[j], kernel_types[k_num])

rand_norm = np.random.normal(0, 1, 1)
u, s, v = np.linalg.svd(C)
print(a.shape)
print(s.shape)
print(u.shape)
z = np.dot(u,np.sqrt(s))*rand_norm

plt.plot(x,z, '.-')
plt.axis([-1,1,-2,2])
plt.show()
