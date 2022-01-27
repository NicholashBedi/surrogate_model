import numpy as np

class ObjectiveFunction:
    def __init__(self, type_i):
        self.type = type_i
        if self.type == "quadratic_1":
            self.limits = [-5, 5]
            self.bounds = [(-5,5), (-5,5)]
            self.optimum = [0,0]
        elif self.type == "quadratic_2":
            self.limits = [-10, 10]
            self.bounds = [(-10,10), (-10,10)]
            self.optimum = [0,0]
        elif self.type == "easoms":
            self.limits = [-10, 10]
            self.bounds = [(-10,10), (-10,10)]
            self.optimum = [np.pi,np.pi]
        else:
            print("Unknown type")
            print(self.type )
    def print_goal(self):
        opt_value = self.wrap_calc(self.optimum, 0)
        print("Optimum should be at: [{:.2f}, {:.2f}] with value {:.2f}".format(
                    self.optimum[0],self.optimum[1],opt_value))
    def wrap_calc(self, x, args):
        noise = args
        return self.calc(np.array(x[0]), np.array(x[1]), noise)

    def calc(self, x, y, noise = 0.1):
        if self.type == "quadratic_1":
            return x**2.0 + y**2.0 + np.random.normal(scale = noise, size = y.shape)
        elif self.type == "quadratic_2":
            return 0.26 * (x**2 + y**2) - 0.48 * x * y + np.random.normal(scale = noise, size = y.shape)
        elif self.type == "easoms":
             return -np.cos(x) * np.cos(y) \
                    * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))
        else:
            print("Unknown type")
            print(self.type )
