import numpy as np
from matplotlib import pyplot as plt

x_range = [0,1.2]
y_range = [-1.6, 2.2]
def test_function(x):
    return -(1.4 - 3*x)*np.sin(18*x)

kernel_types = ["linear", "brownian", "sq_exp", "orn-uhl", "periodic", "symmetric"]
# k_num = 2

def kernel(x,y, type = "linear"):
    if type == kernel_types[0]:
        return np.dot(x.T, y)
    elif type == kernel_types[1]:
        return min(x, y)
    elif type == kernel_types[2]:
        alpha = 0.10
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

def GP(x_a, x_b, y_b, sigma_noise = 0, k_num = 2):
    l = len(x_a)
    k_aa = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            k_aa[i,j] = kernel(x_a[i], x_a[j], type = kernel_types[k_num])
    n_l = len(x_b)
    k_bb = np.zeros((n_l, n_l))
    for i in range(n_l):
        for j in range(n_l):
            k_bb[i,j] = kernel(x_b[i], x_b[j], type = kernel_types[k_num])
    k_ab = np.zeros((l, n_l))
    for i in range(l):
        for j in range(n_l):
            k_ab[i,j] = kernel(x_a[i], x_b[j], type = kernel_types[k_num])
    k_ba = np.zeros((n_l, l))
    for i in range(n_l):
        for j in range(l):
            k_ba[i,j] = kernel(x_b[i], x_a[j], type = kernel_types[k_num])

    c_aa = k_aa + sigma_noise**2*np.eye(l)
    c_bb = k_bb + sigma_noise**2*np.eye(n_l)
    c_bb_inv = np.linalg.inv(c_bb)
    # I dont know what these actually should be
    mu_a = 0
    mu_b = 0

    m = mu_a + k_ab @ c_bb_inv @ (y_b - mu_b)
    d = c_aa - k_ab @ c_bb_inv @ k_ba
    sigma = np.sqrt(np.diag(d))
    return m, sigma


def test_1d():
    x_given = np.array([0, 0.2, 0.4, 0.8, 1.1])
    y_given = test_function(x_given)


    x = np.linspace(x_range[0], x_range[1], 200)
    y_true = test_function(x)


    m, sigma_new = GP(x, x_given, y_given)

    plt.plot(x, y_true, 'b-', label="True Function")
    plt.plot(x_given, y_given, 'o', c = 'b', label='training data')
    plt.plot(x, m, 'r-', label="Predicted Function")
    plt.fill_between(x, m-2*sigma_new, m+2*sigma_new, color='red',
                     alpha=0.15, label='$2 \sigma$')
    plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
    plt.legend(loc="upper left");
    plt.xlabel('x', fontsize=15)
    plt.ylabel('f(x)', fontsize=15)
    plt.show()


if __name__ == "__main__":
    test_1d()
