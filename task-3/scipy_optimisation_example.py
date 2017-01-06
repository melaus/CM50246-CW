# Optimistion example using scipy optimisation tools.
#
# Demonstrates how to optimise the Rosenbrock function using the built in tools
# for a number of different methods. Note that the 'trust-ncg' method requires 
# the Hessian which can be complicated or expensive to compute in many cases so
# you will probably want to use one of the gradient methods for other machine
# learning tasks.
#
# Neill Campbell
# Machine Learning Course, 2016
#

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats
# get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:

def plotOptimFunction(func):
    axisRange = [-3.0, 3.0, -1.0, 3.5]
    resolution = 300

    [xx,yy] = np.meshgrid(np.linspace(axisRange[0], axisRange[1], resolution),         np.linspace(axisRange[2], axisRange[3], resolution))

    # Remember - only want the function eval, not the gradients, etc..
    if (type(func([0.0, 0.0])) is tuple):
        optFunction = lambda x: func(x)[0]
    else:
        optFunction = func

    def f(x, y):
        return optFunction(np.array([x, y]))

    vf = np.vectorize(f)

    R = vf(xx,yy)

    plt.imshow(np.log(R), interpolation='bicubic',
               extent=(axisRange[0], axisRange[1], axisRange[2], axisRange[3]), 
               aspect='auto', origin='lower', cmap='RdBu')


# In[3]:

def rosenbrockFunctionAndGradient(z):
    assert (len(z) == 2)
    a = 1.0
    b = 100.0

    x = z[0]
    y = z[1]

    # The function value
    f = (a - x)**2 + b * ((y - x**2)**2)

    grad_f = np.array([-2.0*(a-x) - 4.0*b*(y-x**2)*x, 2.0*b*(y-x**2)])

    return f, grad_f

def rosenbrockHessian(z):
    assert (len(z) == 2)
    a = 1.0
    b = 100.0

    x = z[0]
    y = z[1]

    hessian_f = np.array([[2.0 - 4.0*b*(y-x**2) + 8.0*b*x**2, -4.0*b*x],        [-4.0*b*x, 2.0*b]])

    return hessian_f


# In[4]:

x0=[-1.0,2.0]

for method in ['trust-ncg', 'L-BFGS-B', 'BFGS']:
    plt.figure()
    plotOptimFunction(rosenbrockFunctionAndGradient)

    def plot_callback(x):
        plt.plot(x[0], x[1], 'kx')

    x_opt = opt.minimize(rosenbrockFunctionAndGradient, x0, method=method, jac=True, hess=rosenbrockHessian, callback=plot_callback)
    print x_opt
    
    plt.plot(x0[0], x0[1], 'bo')
    plt.plot(x_opt.x[0], x_opt.x[1], 'ro')
    
    plt.title('Method: {}, x_opt = [{:.4},{:.4}], Iterations = {}\n'.format(
            method, x_opt.x[0], x_opt.x[1], x_opt.nit))

    plt.show()
