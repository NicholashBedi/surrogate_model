import numpy as np
from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from objective_functions import ObjectiveFunction
import gp_functions as gp_fun


obf = ObjectiveFunction("quadratic_2")
r_min = obf.limits[0]
r_max = obf.limits[1]

xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)
x, y = np.meshgrid(xaxis, yaxis)
results = obf.calc(x, y, noise = 0)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
cs1 = ax1.contourf(x, y, results, cmap='cool')
cbar1 = fig.colorbar(cs1, ax=ax1)

ax1.set_title("True")
ax2.set_title("GP estimate")
ax3.set_title("Error")
ax4.set_title("Uncertanty")
ax5.set_title("EI")
for ax in [ax1,ax2, ax3, ax4, ax5, ax6]:
    ax.set_xlim([r_min, r_max])
    ax.set_ylim([r_min, r_max])

n = 50
np.random.seed(0)
training_data = np.random.uniform(r_min, r_max, (n,2))

# Get guassian process model
# kernels = [1.0 *gp.kernels.RBF(length_scale=1.0),
#             1.0 * gp.kernels.DotProduct(sigma_0=1.0) ** 2]
kernel = gp.kernels.RBF(np.sqrt(1/20), length_scale_bounds = (1e-8, 100))
model = gp.GaussianProcessRegressor(kernel=kernel,
                                    optimizer='fmin_l_bfgs_b',
                                    n_restarts_optimizer=100,
                                    alpha=1e-10,
                                    normalize_y = False,
                                    random_state = 1)

z = obf.calc(training_data[:, 0], training_data[:,1], noise = 0)
model.fit(training_data, z)
mesh_data = np.vstack((x.ravel(), y.ravel())).T
z_pred, pred_std = model.predict(mesh_data, return_std=True)
z_pred = z_pred.reshape(x.shape)
cs2 = ax2.contourf(x,  y, z_pred, levels = cs1.levels, cmap='cool')
cbar2 = fig.colorbar(cs1, ax=ax2)
ax2.scatter(training_data[:,0], training_data[:,1], c='g', alpha = 0.8, s = 1)

error = abs(z_pred - results)
cs3 = ax3.contourf(x,  y, error, cmap='cool')
cbar3 = fig.colorbar(cs3, ax=(ax3))
cs4 = ax4.contourf(x, y, pred_std.reshape(x.shape), cmap='cool')
cbar4 = fig.colorbar(cs4, ax=(ax4))


new_sample, EI = gp_fun.EI_learning(mesh_data, z_pred.ravel(), pred_std)
cs5 = ax5.contourf(x, y, EI.reshape(x.shape), cmap='cool')
ax5.scatter(new_sample[0], new_sample[1], c='g', s = 2)
cbar5 = fig.colorbar(cs5, ax=(ax5))


plt.show()
