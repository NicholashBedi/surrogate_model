import numpy as np
from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from objective_functions import ObjectiveFunction
import gp_functions as gp_fun
import matplotlib.animation as animation
import gp_inference as gpi

class MultiDimensionGP:
    def __init__(self, type= "quadratic_1", noise_i = 0.1):
        self.noise = noise_i
        self.obf = ObjectiveFunction(type)
        xaxis = np.arange(self.obf.limits[0], self.obf.limits[1], 0.1)
        yaxis = np.arange(self.obf.limits[0], self.obf.limits[1], 0.1)
        self.x, self.y = np.meshgrid(xaxis, yaxis)
        self.mesh_data = np.vstack((self.x.ravel(), self.y.ravel())).T

    def on_pause(self, x):
        self.anim.resume()

    def set_initial_training_data(self):
        n = 20
        np.random.seed(0)
        self.train_input_data = np.random.uniform(self.obf.limits[0],
                                                self.obf.limits[1],
                                                (n,2))
        self.train_objective = self.obf.calc(self.train_input_data[:, 0],
                                            self.train_input_data[:,1],
                                            noise = self.noise)
        return

    def create_GP_model(self):
        kernel = gp.kernels.RBF(1,
                            length_scale_bounds = (1e-2, 100))
        # kernel = gp.kernels.RBF(10,
        #                     length_scale_bounds = "fixed")
        self.model = gp.GaussianProcessRegressor(kernel=kernel,
                                            optimizer='fmin_l_bfgs_b',
                                            n_restarts_optimizer=100,
                                            alpha=1e-10,
                                            normalize_y = False,
                                            random_state = 1)

    def set_up_graphs(self):
        self.ax0.set_title("True")
        self.ax1.set_title("GP estimate")
        self.ax2.set_title("Error")
        self.ax3.set_title("Uncertanty")
        self.ax4.set_title("EI")
        self.ax5.set_title("EI/Cost")
        self.cs = []
        self.axes = [self.ax0, self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]
        self.cbar = []
        for ax in self.axes:
            zero = np.zeros((self.x.shape))
            self.cs.append(ax.contourf(self.x, self.y, zero, cmap='cool'))
            ax.set_xlim([self.obf.limits[0], self.obf.limits[1]])
            ax.set_ylim([self.obf.limits[0], self.obf.limits[1]])
            self.cbar.append(self.fig.colorbar(self.cs[0], ax=ax))

        # Graph 1 stays constant
        self.z = self.obf.calc(self.x, self.y, noise = 0)
        self.cs[0] = self.ax0.contourf(self.x, self.y, self.z, cmap='cool')

        self.ax1.scatter([], [], c='g', alpha = 0.8, s = 1) # Previously Tested points
        self.ax4.scatter([], [], c='g', s = 2) # Points to test
        return

    def main(self, frame):
        # GP estimate
        use_sklearn = True
        if use_sklearn:
            self.model.fit(self.train_input_data, self.train_objective)
            z_pred, pred_std = self.model.predict(self.mesh_data, return_std=True)
        else:
            z_pred, pred_std = gpi.GP(self.mesh_data, self.train_input_data,
                                self.train_objective, sigma_noise = self.noise)
        z_pred = z_pred.reshape(self.x.shape)
        self.cs[1] = self.ax1.contourf(self.x,  self.y, z_pred,
                        levels = self.cs[0].levels, cmap='cool')
        self.cbar[1].remove()
        self.cbar[1] = self.fig.colorbar(self.cs[1], ax=self.ax1)
        self.ax1.scatter(self.train_input_data[:,0],
                        self.train_input_data[:,1], c='g', alpha = 0.8, s = 1)
        # Error plot
        error = abs(z_pred - self.z)
        self.cs[2] = self.ax2.contourf(self.x,  self.y, error, cmap='cool')
        self.cbar[2].remove()
        self.cbar[2] = self.fig.colorbar(self.cs[2], ax=(self.ax2))
        # Uncertanty
        self.cs[3] = self.ax3.contourf(self.x,  self.y,
                                pred_std.reshape(self.x.shape), cmap='cool')
        self.cbar[3].remove()
        self.cbar[3] = self.fig.colorbar(self.cs[3], ax=(self.ax3))
        # EI
        new_sample, EI = gp_fun.EI_learning(self.mesh_data, z_pred.ravel(), pred_std)
        print(new_sample)
        self.cs[4] = self.ax4.contourf(self.x,  self.y, EI.reshape(self.x.shape), cmap='cool')
        self.ax4.scatter(new_sample[0], new_sample[1], c='g', s = 4)
        self.cbar[4].remove()
        self.cbar[4] = self.fig.colorbar(self.cs[4], ax=(self.ax4))

        # Add training data
        self.train_input_data = np.append(self.train_input_data,
                                            new_sample.reshape(1,2), axis = 0)
        self.train_objective = np.append(self.train_objective,
                                [self.obf.calc(new_sample[0], new_sample[1],
                                            noise = self.noise)])
        self.anim.pause()
        return

    def animation(self):
        self.fig, ((self.ax0, self.ax1), (self.ax2, self.ax3),
                    (self.ax4, self.ax5)) = plt.subplots(3,2)
        self.fig.canvas.mpl_connect('button_press_event', self.on_pause)
        self.create_GP_model()
        self.set_up_graphs()
        self.set_initial_training_data()
        self.anim = animation.FuncAnimation(self.fig, self.main,
                        frames=10, interval = 100, blit=False)
        plt.show()

a = MultiDimensionGP()
a.animation()
