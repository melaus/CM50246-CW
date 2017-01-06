import numpy as np
import matplotlib.pyplot as plt

def plot_classification_data(X, Y, true_labels=None):
    '''plot_classification_data plots classification data for use with 
        supervised learning classification tasks.

        plot_classification_data(X, Y, true_labels)
     
        X = Matrix of [NumberOfTrainingPoints x NumInputDimensions] size  
            containing input data
        Y = Matrix of [NumberOfTrainingPoints x NumClasses] size
            containing the probability of each point belonging to the
            corresponding class e.g. [N x 2] for binary problems
        true_labels = Vector of [NumberOfTrainingPoints] with a zero
            based integer of the true class of each variable used to 
            show which points are classified correctly (optional)

    Machine Learning Course
    Neill D.F. Campbell, 2016
    '''

    [N, Q] = X.shape

    assert (Q == 2)
    assert (Y.shape[0] == N)

    assert (Y.ndim == 2)

    # Number of classes..
    K = Y.shape[1]
    assert (K > 1)

    if true_labels is not None:
        assert (np.min(true_labels) >= 0)
        assert (np.max(true_labels) < K)

    if K == 2:
        for n,c in enumerate(Y):
            plt.scatter(X[n,0], X[n,1], color=np.r_[c, [0.0,1.0]], 
                        s=50, edgecolors='k')
    elif K == 3:
        for n,c in enumerate(Y):
            plt.scatter(X[n,0], X[n,1], color=np.r_[c, [1.0]], 
                        s=50, edgecolors='k')
    else:
        if true_labels is not None:
            plt.scatter(X[:,0], X[:,1], c=true_labels / K, 
                        cmap='gist_rainbow', s=50, edgecolors='k')
        else:
            plt.scatter(X[:,0], X[:,1], c=np.argmax(Y, axis=1) / K,
                        cmap='gist_rainbow', s=50, edgecolors='k')
    
    if true_labels is not None:
        bad = np.argmax(Y,axis=1) != true_labels
        plt.scatter(X[bad,0], X[bad,1], s=90, color=[0.0, 0.0, 0.0, 0.0], edgecolors='r')

    plt.axis('square')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.grid(True)


def generate_classification_data(test_case=None, num_training_points=60, num_classes=2, 
    num_input_dimensions=2, plot_data=True):
    '''
    Generates Data for use with supervised learning regression tasks.

    ******************************************************************
    NOTE: Please run on all three test_case datasets!
    ******************************************************************

    X, true_labels = generate_classification_data(test_case,
                                                  num_training_points, 
                                                  num_classes, 
                                                  num_input_dimensions, 
                                                  plot_data)

        test_case = integer in range [0,1,2] which specifies the test case
            provide data for
        num_training_points = how many training points to generate (integer)
        num_classes = how many classes? (two for binary) (integer > 1)
        num_input_dimensions = how many input dimensions to use (integer)
        plot_data = will plot data in a figure if set to true (true/false)

        X = Matrix of [num_training_points x num_input_dimensions] size  
           containing input data
        true_labels = Vector of [num_training_points] with the integer class
            of the corresponding training point (zero based)

      For an example just run 

        generate_classification_data()

      without parameters to get the default settings.

    Machine Learning Course
    Neill D.F. Campbell, 2016
    '''

    if test_case is None:
        for t in range(3):
            generate_classification_data(t, num_training_points, 
                                         num_classes, num_input_dimensions, 
                                         plot_data)
        plt.show()
        return None, None
    else:
        assert (type(test_case) == int)
        assert (test_case >= 0)
        assert (test_case <= 2)
    
    assert (num_training_points % num_classes == 0)

    if (num_input_dimensions != 2):
        print('WARNING: Setting num_input_dimensions = 2!')
        num_input_dimensions = 2

    N_K = int(num_training_points / num_classes)

    X = np.zeros([num_training_points, num_input_dimensions])
    true_labels = np.zeros(num_training_points, dtype=int)
    
    np.random.seed(0)
    
    if test_case == 0:
        phi = np.linspace(0.0, 2.0*np.pi, num_classes+1)
        for k in range(num_classes):
            X[k*N_K:(k+1)*N_K,:] = 0.5 * np.array([np.cos(phi[k]), np.sin(phi[k])]) + \
                (0.3 / num_classes) * np.random.randn(N_K, num_input_dimensions)
            true_labels[k*N_K:(k+1)*N_K] = k * np.ones(N_K, dtype=int)
            
    elif test_case == 1:
        phi = np.linspace(0.0, 2.0*np.pi, 2*(num_classes+1))
        N_K_a = int(np.ceil(N_K / 2.0))
        N_K_b = N_K - N_K_a
        for k in range(num_classes):
            X[k*N_K:(k*N_K+N_K_a),:] = 0.5 * np.array([np.cos(phi[k]), np.sin(phi[k])]) + \
                (0.3 / num_classes) * np.random.randn(N_K_a, num_input_dimensions)
            X[(k*N_K+N_K_a):(k+1)*N_K,:] = 0.5 * np.array([np.cos(np.pi+phi[k]), np.sin(np.pi+phi[k])]) + \
                (0.3 / num_classes) * np.random.randn(N_K_b, num_input_dimensions)
            true_labels[k*N_K:(k+1)*N_K] = k * np.ones(N_K, dtype=int)
    
    elif test_case == 2:
        phi = np.linspace(1.0, 2.0*np.pi, N_K)
        offset_theta = np.linspace(0.0, np.pi, num_classes)
        a = 0.5 / np.pi

        for k,t in enumerate(offset_theta):
            X[k*N_K:(k+1)*N_K,:] = a * phi[:,np.newaxis] * np.c_[np.cos(t + phi), np.sin(t + phi)]
            true_labels[k*N_K:(k+1)*N_K] = k * np.ones_like(phi, dtype=int)

        X = X + (0.1 / num_classes) * np.random.randn(num_training_points, num_input_dimensions)

    Y = np.zeros([num_training_points, num_classes])
    Y[np.arange(num_training_points), true_labels] = 1
        
    if plot_data:
        plt.figure()
        plot_classification_data(X, Y, true_labels=true_labels)
        plt.title('Training data for Test Case {}'.format(test_case))

    return X, true_labels