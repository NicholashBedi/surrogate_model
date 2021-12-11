from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

def test_linear_svm():
    n = 8
    a_data = np.random.uniform(0,10, (n,2))
    a_data[:, 1] = a_data[:, 0]*3 - 4 + np.random.uniform(-1,1, (n))

    b_data = np.random.uniform(0,10, (n,2))
    b_data[:, 1] = b_data[:, 0]*3 + 5 + np.random.uniform(-1,1, (n))

    x = np.concatenate((a_data,b_data))
    y = np.concatenate((np.ones(n), np.ones(n)*-1))

    lin = svm.LinearSVC(max_iter=100000)
    result = lin.fit(x,y)
    print(result.get_params())
    print(result.coef_)
    print(result.intercept_)
    print(result.n_iter_)

    b = result.intercept_
    w = result.coef_[0]

    x_line = np.linspace(0, 10, 100)
    y_line = (-b - w[0]*x_line)/w[1]
    plt.scatter(a_data[:, 0], a_data[:,1], c = 'r')
    plt.scatter(b_data[:, 0], b_data[:,1], c = 'b')
    plt.plot(x_line, y_line)
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def get_iris_data():
    iris = datasets.load_iris()
    return iris.data[:,:2], iris.target

def plot_svm(x,y,svm_results):
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    xx, yy = make_meshgrid(x[:,0], x[:, 1])
    Z = svm_results.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor="k")
    plt.show()

def test_linear_multi_class():
    x, y = get_iris_data()
    # lin = svm.LinearSVC(max_iter=100000)
    # lin = svm.SVC(kernel="rbf", gamma=10, C=1)
    lin = svm.SVC(kernel="poly", degree=5, gamma="auto", C=1)
    result = lin.fit(x,y)
    plot_svm(x,y,result)


test_linear_multi_class()

# test_linear_svm()
