import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

def plot3(a, b, c, mark='o', col='r'):
    pylab.ion()
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(a, b, c, marker=mark, color=col)
    return ax

def plot_regression_data(X, Y, equation=None, noise=None):
    '''plot_regression_data plots regresiion data for use with supervised learning 
        regression tasks.

        plot_regression_data(X, Y, equation, noise)
     
        X = Matrix of [NumberOfTrainingPoints x NumInputDimensions] size  
            containing input data
        Y = Matrix of [NumberOfTrainingPoints x 1] size
            containing output data
        equation = Coefficients of the polynomial used to evaluate the
            function d before noise. Thus the true function is
            y_true = sum(polyval(trueEquation, x), 2)
        noise = std. of the noise if error bars are required.

    Machine Learning Course
    Neill D.F. Campbell, 2016
    '''

    [N, Q] = X.shape

    assert (Q <= 2)
    assert (Y.shape[0] == N)
    # print(Y.shape)

    if Y.ndim > 1:
        assert (Y.shape[1] == 1)

    if (Q == 1):
        plt.plot(X, Y, 'rx')
        if equation is not None:
            num_samples = 250
            xx = np.linspace(np.min(X), np.max(X), num_samples)
            f = np.polyval(equation, xx)
            plt.plot(xx, f, 'r-')
            plt.grid()

            if noise is not None:
                plt.plot(xx, f + 2.0*noise, 'r:')
                plt.plot(xx, f - 2.0*noise, 'r:')
    else:
        ax = plot3(X[:,0], X[:,1], Y, 'o', 'r')
        if equation is not None:
            num_samples = 250
            xx, yy = np.meshgrid(np.linspace(np.min(X[:,0]), np.max(X[:,0]), num_samples), \
                np.linspace(np.min(X[:,1]), np.max(X[:,1]), num_samples))
            zz = np.c_[xx.flatten(), yy.flatten()]
            f = np.sum(np.polyval(equation, zz), axis=1)
            f = np.reshape(f, [num_samples,num_samples])
            surf = ax.plot_surface(xx, yy, f, cmap=plt.cm.jet, antialiased=True)


def generate_regression_data(num_training_points=50, polynomial_order_of_data=1, 
    signal_to_noise_ratio=0.1, num_input_dimensions=1, plot_data=True):
    '''
    Generates Data for use with supervised learning regression tasks.

    X, Y, true_equation, noise_std = generate_regression_data(
                                                      num_training_points, 
                                                      polynomial_order_of_data, 
                                                      signal_to_noise_ratio, 
                                                      num_input_dimensions, 
                                                      plot_data)

        num_training_points = how many training points to generate (integer)
        polynomial_order_of_data = order of the underlying polynomial (integer > 0)
        signal_to_noise_ratio = (0.0 - 1.0) ratio of noise (float)
        num_input_dimensions = how many input dimensions to use (integer)
        plot_data = will plot data in a figure if set to true (true/false)

        NOTE: Details about cell arrays available at:
           https://yagtom.googlecode.com/svn/trunk/html/dataStructures.html#13

        X = Matrix of [NumberOfTrainingPoints x NumInputDimensions] size  
           containing input data
        Y = Matrix of [NumberOfTrainingPoints x 1] size
           containing output data
        true_equation = Coefficients of the polynomial used to evaluate the
           function d before noise. Thus the true function is
           y_true = np.sum(np.polyval(trueEquation, x), axis=2)
        noise_std = std deviation of the noise applied

      For an example just run 

        X, Y, true_equation, noise_std = generate_regression_data()

      without parameters to get some default settings.


      Commands to plot a 2D distribution and data:

            X, Y, true_equation, noise_std = generate_regression_data()
            plt.figure()
            plot_regression_data(X, Y, true_equation, noise_std)
            plt.show()

    Machine Learning Course
    Neill D.F. Campbell, 2016
    '''


    if (num_input_dimensions > 2) and (plot_data == True):
        print('WARNING! Can only plot data for num_dimensions <= 2')
        plot_data = False

    def isinteger(x):
        return np.equal(np.mod(x, 1), 0)

    assert (isinteger(num_training_points))
    assert (isinteger(polynomial_order_of_data))
    assert (isinteger(num_input_dimensions))

    assert ((polynomial_order_of_data >= 1) and (polynomial_order_of_data < 10))

    assert (num_training_points > 0)
    assert (num_input_dimensions > 0)

    np.random.seed(0)

    X = np.random.rand(num_training_points, num_input_dimensions) - 0.5

    true_equation = 0.5 * np.random.randn(polynomial_order_of_data+1, 1)

    Y = np.sum(np.polyval(true_equation, X), axis=1, keepdims=True)

    # print(Y, Y.shape)

    true_equation[-1] = true_equation[-1] - np.mean(Y)

    # print(X, true_equation, np.polyval(true_equation, X))

    Y = np.sum(np.polyval(true_equation, X), axis=1, keepdims=True)

    noise_std = np.std(Y) * signal_to_noise_ratio

    # print(np.std(Y), Y.shape)

    Y = Y + noise_std * np.random.randn(num_training_points, 1)

    # print(noise_std * np.random.randn(num_training_points, 1))

    if (plot_data):
        plt.figure()
        plot_regression_data(X, Y, true_equation, noise_std)
        plt.title('Original Data')
        plt.savefig('write-up/tmp/original-data.png', dpi=500)
        plt.show()

    return X, Y, true_equation, noise_std


