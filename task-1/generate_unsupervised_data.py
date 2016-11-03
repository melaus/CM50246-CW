import numpy as np
import matplotlib.pyplot as plt


def plot_covar_ellipse(mu, sigma, color, alpha=1.0):
    '''plotCovarEllipse plots a Gaussian covariance ellipse on a 2D plot.

        plotCovarEllipse(mu, sigma, style)
     
        mu = 2 x 1 mean vector
        sigma = 2 x 2 positive definite covariance matrix

        style = optional plotting style commands (can be ignored)

    Machine Learning Course
    Neill D.F. Campbell, 2016
    '''

    L = np.linalg.cholesky(sigma)

    t = np.linspace(0, 2.0*np.pi, 50)
    x = np.c_[np.cos(t),np.sin(t)]

    x = np.dot(L, x.T).T + mu

    plt.plot(mu[0], mu[1], '{}o'.format(color), alpha=alpha)
    plt.plot(x[:,0], x[:,1], '{}-'.format(color), alpha=alpha)


class gmm_distribution(object):
    '''
    Contains a set of Gaussian Mixture Model parameters.

        distribution.weight_n = array of weights (scalar)
        distribution.mu_n = array of means (NumDims x 1)
        distribution.sigma_n = array of covariances (NumDims x NumDims)
        distribution.NumClusters
        distribution.NumDimensions

    Machine Learning Course
    Neill D.F. Campbell, 2016
    '''
    def __init__(self, number_of_components, number_of_dimensions):
        self.NumClusters = number_of_components
        self.NumDimensions = number_of_dimensions

        self.weight_n = np.ones([number_of_components,]) * (1.0 / number_of_components)
        self.mu_n = [np.zeros(number_of_dimensions,)] * number_of_components
        self.sigma_n = [np.eye(number_of_dimensions)] * number_of_components

    def plot(self):
        '''
        Plots the current set of Gaussian Mixture Model parameters.
        '''
        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        for n in range(self.NumClusters):
            plot_covar_ellipse(self.mu_n[n], self.sigma_n[n], colors[n % len(colors)])
            plt.text(self.mu_n[n][0], self.mu_n[n][1], \
                '  {}'.format(self.weight_n[n]), color=colors[n % len(colors)])

    def sample_points(self, num_points):
        '''
        Samples a number of points from the GMM distribution.
        '''
        N = np.floor(self.weight_n * num_points)
        N = N.astype(int)

        if (not N.sum() == num_points):
            N[0] = N[0] - (N.sum() - num_points)

        assert (np.all(N > 2))
        assert (N.sum() == num_points)

        cN = np.r_[0, np.cumsum(N)]

        x = np.zeros((self.NumDimensions, num_points))

        for n in range(self.NumClusters):
            L = np.linalg.cholesky(self.sigma_n[n])
            x[:,cN[n]:cN[n+1]] = L.dot(np.random.randn(self.NumDimensions, N[n])) + np.c_[self.mu_n[n]]

        return x


def generate_unsupervised_data(num_dimensions=2, num_clusters=5, num_points=500, plot_data=True):
    '''
    Generates Data for use with K-Means clustering or Gaussian Mixture Model fitting.

    x, true_distribution = generate_unsupervised_data(num_dimensions, 
                                                      num_clusters, 
                                                      num_points, 
                                                      plot_data)

      num_dimensions = how many dimensions to use (integer)
      num_clusters = how many clusters to use (integer)
      num_points = how many data points to sample (this must be at least 5
          times the number of clusters) (integer)
      plot_data = will plot data in a figure if set to 'True' and the 
          number of dimensions is 2 (True/False)

      x = Matrix of [num_dimensions x num_points] size containing the data

      true_distribution = Structure containing the real GMM parameters used
          to generate the data (for comparison with algorithmic results).

          trueDistribution.weight_n = cell array of weights (scalar)
          trueDistribution.mu_n = cell array of means (NumDims x 1)
          trueDistribution.sigma_n = cell array of covariances (NumDims x NumDims)
          trueDistribution.NumClusters
          trueDistribution.NumDimensions

      For an example just run 

        x, true_distribution = generate_unsupervised_data()

      without parameters to get some default settings.


      Commands to plot a 2D distribution and data:

            x, true_distribution = generate_unsupervised_data()
            plt.figure()
            plt.plot(x[:,0], x[:,1], 'kx')
            true_distribution.plot()

    Machine Learning Course
    Neill D.F. Campbell, 2016
    '''

    if (not num_dimensions == 2):
        print('WARNING! Can only plot data for num_dimensions = 2')

    def isinteger(x):
        return np.equal(np.mod(x, 1), 0)

    assert (isinteger(num_dimensions))
    assert (isinteger(num_clusters))
    assert (isinteger(num_points))

    assert (num_clusters > 0)
    assert (num_clusters < 0.2 * num_points)
    assert (num_dimensions > 0)

    np.random.seed(0)

    d = gmm_distribution(num_clusters, num_dimensions)

    d.weight_n = np.random.dirichlet(np.ones(num_clusters)*10.0)
    d.mu_n = [np.random.randn(num_dimensions) for i in range(num_clusters)]

    def get_rand_sigma():
        A = np.random.randn(num_dimensions,num_dimensions)
        d, Q = np.linalg.eig(A.dot(A.T))
        D = np.diag(0.1 * np.random.rand(num_dimensions))
        return Q.dot(D).dot(Q.T)

    d.sigma_n = [get_rand_sigma() for i in range(num_clusters)]

    x = d.sample_points(num_points)

    if (plot_data):
        plt.figure()
        plt.plot(x[0,:], x[1,:], 'kx')
        d.plot()
        plt.title('Original Data')
        plt.draw()

    return x, d


