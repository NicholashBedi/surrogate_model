# import all packages and set plots to be embedded inline
import numpy as np
import scipy.stats as st
from scipy.integrate import quad
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.set_printoptions(precision=2,suppress=True)


def test_function(x):
    return (x*6-2)**2*np.sin(x*12-4)
# Test function values
x_linspace = np.linspace(0, 1, 100)
y_linspace = test_function(x_linspace)

def EI_learning(candidates, y_pred, pred_std):
    current_objective = y_pred[np.argmin(y_pred)]

    EI = (current_objective-y_pred)*st.norm.cdf((current_objective-y_pred)/pred_std) \
            +pred_std*st.norm.pdf((current_objective-y_pred)/pred_std)

    new_sample = candidates[np.argmax(EI)]

    return new_sample, EI

def improve_model(model, candidates, x_train, y_train, colour_train, max_iter = 50):
    colour_train = np.zeros(x_train.shape[0])
    EI = np.array([10])
    num_iter = 0
    y_predictions = []
    pred_std = []
    while(np.max(EI) > 0.0001 and num_iter < max_iter):
        model.fit(x_train.reshape(-1, 1), y_train,)
        temp_y_pred, temp_pred_std = model.predict(candidates.reshape((-1, 1)), return_std=True)
        y_predictions.append(temp_y_pred)
        temp_pred_std += 1e-8
        pred_std.append(temp_pred_std)
        new_sample, EI = EI_learning(candidates, temp_y_pred, temp_pred_std)
        # plot_results(x_train, y_train, colour_train, y_pred, pred_std, new_sample)
        x_train = np.append(x_train, new_sample)
        y_train = np.append(y_train, test_function(new_sample))
        colour_train = np.append(colour_train, colour_train[-1] + 1)
        num_iter += 1
    return x_train, y_train, colour_train, y_predictions, pred_std


# Initial training data
x_train = np.array([0, 0.1, 0.5, 0.9, 1])
initial_length = len(x_train)
y_train = test_function(x_train)
colour_train = np.zeros(x_train.shape[0])
# Train initial Gaussian Process (GP) model
kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(1.0, (1e-8, 1e8))
model = gp.GaussianProcessRegressor(kernel=kernel,
                                    optimizer='fmin_l_bfgs_b',
                                    n_restarts_optimizer=30,
                                    alpha=1e-10,
                                    normalize_y=True)
x_train, y_train, colour_train, y_predictions, pred_std = improve_model(model, x_linspace, x_train, y_train, colour_train)

fig = plt.figure()
ax = plt.axes(xlim=(-0.1,1.1), ylim=(-10,20))
true_func_line, = ax.plot(x_linspace, y_linspace, 'r--', label="True Function")
# Display global minimum
min_index = np.argmin(y_linspace)
ax.plot(x_linspace[min_index], y_linspace[min_index],'ro', markerfacecolor='r', label='Global Minimum')

predict_func_line, = ax.plot([], [], 'b--', label="Predicted Function")
training_data_plt, = ax.plot([], [], 'o', c = 'b', label='training data')
initial_data_plt, = ax.plot(x_train[:initial_length], y_train[:initial_length],
                            'o', c = 'g', label='Initial data')
area_infill = ax.fill_between( [], [], [], alpha = 0.5, color='k')
new_sample, = ax.plot([], [], c = 'g', label='Next sample')
ax.legend(loc="upper left");
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('f(x)', fontsize=15)

def init():
    predict_func_line.set_data([],[])
    training_data_plt.set_data([],[])
    new_sample.set_data([],[])
    return predict_func_line, training_data_plt, new_sample,

def animate(i):
    predict_func_line.set_data(x_linspace, y_predictions[i])
    training_data_plt.set_data(x_train[initial_length:initial_length+i],
                                y_train[initial_length:initial_length+i])
    area_infill = ax.fill_between(x_linspace,
                        y_predictions[i] - 1.9600 * pred_std[i],
                        y_predictions[i] + 1.9600 * pred_std[i], alpha = 0.5, color='k')
    # new_sample.set_data([-0.5,-0.5],[-10, 20])
    new_sample.set_data([x_train[initial_length+i], x_train[initial_length+i]], [-10, 20])
    return predict_func_line, training_data_plt, new_sample, area_infill

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(y_predictions),
                        interval = 1000, blit=True)
plt.show()
