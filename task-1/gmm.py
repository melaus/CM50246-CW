import argparse
from generate_unsupervised_data import generate_unsupervised_data, plot_covar_ellipse
from random import sample
import numpy as np
from numpy.linalg import det, inv
from math import exp
from itertools import izip
import matplotlib.pyplot as plt
from kmeans import kmeans


def initialise_parameters(args_p, args_c, args_kmeans, pts):
    # print 'initialising parameters...'
    samples = [np.array(sample(pts, args_p / args_c)).T for k in range(args_c)]
    weights = np.array([1.0/args_c for k in range(args_c)])
    means = np.array(sample(pts, args_c))
    if args_kmeans:
        _, means = kmeans(pts, means)
        means = means.T
    covars = np.array([np.cov(group_k) for group_k in samples])

    # print weights
    # print means
    # print covars

    return weights, means, covars


def norm_dist_pdf(dim, x, mean, covar):
    """
    normal distribution likelihood based on inputs
    """
    diff = np.mat(x - mean)
    # print '>>>>> norm_dist_pdf'
    # print 'part1:', 1/((2*np.pi)**(dim/2))
    # print 'part2:', np.power(det(covar), -0.5)
    # print 'part3:', exp(-0.5*diff*inv(covar)*diff.T)
    # print '<<<<<<'
    return 1/((2*np.pi)**(dim/2)) * np.power(det(covar), -0.5) * exp(-0.5*diff*inv(covar)*diff.T)


def log_likelihood(dim, all_reps, weights, means, covars):
    """
    Calculate the log likelihood of the GMM
    :param: dim: int. Dimensions of a datapoint
    :param all_reps: list of pairs in the form (pt, [responsibilities for all groups]).
    :return:
    """

    weighted = [
        [
            weight * norm_dist_pdf(dim, pt, mean, covar)
            for (rep, weight, mean, covar) in izip(reps, weights, means, covars)
        ]
        for (pt, reps) in all_reps]

    # print np.sum(np.log([sum(cluster) for cluster in weighted]))

    likelihood = np.sum(np.log([sum(cluster) for cluster in weighted]))
    return likelihood


def gmm(dim, pts, weights, means, covars, title):
    """
    Fitting GMM using EM
    :param dim: int. Dimension of datapoints
    :param k: int. Number of clusters
    :param pts: np.array. Datapoints in [[x,y]...] format
    :param weights: np.array. A 1D list of weights
    :param means: np.array. A 2D list of means
    :param covars: np.array. A list of (2,2) covars
    :return:
    """
    # print '==================== GMM start ===================='

    # initial parameter assignments
    cur_weights, cur_means, cur_covars = weights, means, covars
    cur_log_likelihood = 0

    i = 1
    try:
        while True:
            # print '\n\n---------- Iteration {0} ----------'.format(i)
            all_reps, max_reps = e_step(pts, dim, cur_weights, cur_means, cur_covars)
            new_weights, new_means, new_covars = m_step(all_reps)

            new_log_likelihood = log_likelihood(dim, all_reps, new_weights, new_means, new_covars)
            # print 'log_likelihood:', new_log_likelihood
            #
            # print 'cur_means:'
            # print cur_means
            # print ''
            # print 'new_means:'
            # print new_means

            if (cur_means == new_means).all() or new_log_likelihood == cur_log_likelihood or i == 10000:
                # print 'new_log_likelihood == cur_log_likelihood'
                plot_outcome(pts.T, new_means.T, new_covars, title)
                print 'iterations: {0}'.format(i)
                return
            else:
                cur_weights, cur_means, cur_covars = new_weights, new_means, new_covars
                cur_log_likelihood = new_log_likelihood
                plt.figure(2)
                plot_covar_ellipse(cur_means.T, cur_covars, 'b', alpha=0.3)
            i += 1
    except:
        print 'killed/ broke at iteration {0}'.format(i)


def e_step(pts, dim, weights, means, covars):
    """
    EXPECTATION step to calculate responsibilities
    :param pts:
    :param dim:
    :param weights:
    :param means:
    :param covars:
    :return:
    """
    # print 'in e_step...'
    # iterate through all points
    max_reps = []
    all_reps = []

    for pt in pts:
        reps = np.array([weight * norm_dist_pdf(dim, pt, mean, covar) for weight, mean, covar in izip(weights, means, covars)])
        reps = reps/np.sum(reps)

        # (point, cluster_number, responsibility)
        # max_reps += [(pt, np.argmax(reps), np.max(reps))]
        all_reps += [(pt, reps)]

    return all_reps, max_reps


def m_step(all_r_k):
    """
    :param all_r_k: list. (pt, responsibilities)
    """
    sum_all_r_k = sum(map(lambda info: info[1], all_r_k))
    new_weights = sum_all_r_k / np.sum(sum_all_r_k)
    new_means = sum(np.array([[ r_info[0] * r_k for r_k in r_info[1] ] for r_info in all_r_k])) / sum_all_r_k[:, None]
    # print 'new_means:\n{0}'.format(new_means)

    new_covars_raw = np.array([
                               [
                                r * np.mat((r_info[0] - mean)).T * np.mat((r_info[0] - mean))
                                for r, mean in izip(r_info[1], new_means)
                               ]
                               for r_info in all_r_k])

    # print 'new_covars_raw.shape: {0}'.format(new_covars_raw.shape)

    new_covars = np.array([c / sum_r for c, sum_r in izip(sum(new_covars_raw), sum_all_r_k)])
    # print 'new_covars:\n{0}'.format(new_covars)

    return new_weights, new_means, new_covars


def plot_outcome(data, means, covars, title):
    plt.figure(2)
    plt.plot(data[0, :], data[1, :], 'kx')
    # plt.plot(centroids[0, :], centroids[1, :], 'rx')
    plot_covar_ellipse(means, covars, 'r')
    plt.title(title)


def main():
    args = get_args()
    print args

    pts, true_dist = generate_unsupervised_data(num_dimensions=args.d,
                                                num_clusters=args.k,
                                                num_points=args.p,
                                                plot_data=True)

    weights, means, covars = initialise_parameters(args.p, args.c, args.kmeans, pts.T)
    # weights, means, covars = initialise_parameters(args, pts.T)

    gmm(args.d, pts.T, weights, means, covars, args.t)
    plt.show()


def get_args():
    """
    get CLi arguments
    :return: argparse
    """
    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('--d', '-num_dimensions', type=int, default=2, help='the number of dimensions in generated data')
    parser.add_argument('--k', '-num_clusters', type=int, default=5, help='the number of clusters')
    parser.add_argument('--c', '-num_centroids', type=int, default=5, help='the number of clusters')
    parser.add_argument('--p', '-num_points', type=int, default=500, help='the number of clusters in generated data')
    parser.add_argument('--kmeans', '-kmeans', action='store_true', default=False, help='initialise with k-means')
    parser.add_argument('--t', '-title', type=str, default='GMM', help='title of plot')

    return parser.parse_args()


if __name__ == '__main__':
    main()

