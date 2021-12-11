import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

n = 4
a_data = np.random.uniform(0,10, (n,2))
a_data[:, 1] = a_data[:, 0]*3 - 4 + np.random.uniform(-1,1, (n))

b_data = np.random.uniform(0,10, (n,2))
b_data[:, 1] = b_data[:, 0]*3 + 5 + np.random.uniform(-1,1, (n))


def to_optimize(alpha, x, y):
    val = alpha.sum()
    # print(val)
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            adjust = 0.5*alpha[i]*alpha[j]*y[i]*y[j]*np.dot(x[i], x[j])
            # print("i: {}, j: {}, adjust: {}".format(i, j , adjust))
            val -= adjust
    # I don't know why I multiple by -1???
    return -1*val

alpha_0 = np.ones((2*n))
x = np.concatenate((a_data,b_data))
y = np.concatenate((np.ones(n), np.ones(n)*-1))
bnds = [(0, None) for x in range(2*n)]
# print(alpha_0)
# print(x)
# print(y)
# print(to_optimize(alpha_0, x, y))
optimize_results = scipy.optimize.minimize(to_optimize, alpha_0, args = (x, y), bounds=bnds)
print(optimize_results)
alpha_optimal = optimize_results.x
# get normal vector
w = np.zeros(2)
for i in range(2*n):
    w += alpha_optimal[i]*y[i]*x[i]
print(w)

# get b
b = 0
for i in range(2*n):
    if alpha_optimal[i] > 0.001:
        b = 1/y[i] - np.dot(w, x[i])
        print(i)
        break
print(b)

x_line = np.linspace(0, 10, 100)
y_line = (-b - w[0]*x_line)/w[1]
plt.scatter(a_data[:, 0], a_data[:,1], c = 'r')
plt.scatter(b_data[:, 0], b_data[:,1], c = 'b')
plt.plot(x_line, y_line)
plt.show()
