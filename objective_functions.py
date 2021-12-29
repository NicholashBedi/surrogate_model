import numpy as np

class ObjectiveFunction:
    def __init__(self, type_i):
        self.type = type_i
        if self.type == "quadratic_1":
            self.limits = [-5, 5]
        elif self.type == "quadratic_2":
            self.limits = [-10, 10]
        else:
            print("Unknown type")
            print(self.type )
    def calc(self, x, y, noise = 0.1):
        if self.type == "quadratic_1":
            return x**2.0 + y**2.0 + y + np.random.normal(scale = noise, size = y.shape)
        elif self.type == "quadratic_2":
            return 0.26 * (x**2 + y**2) - 0.48 * x * y + np.random.normal(scale = noise, size = y.shape)
        else:
            print("Unknown type")
            print(self.type )
