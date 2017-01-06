import numpy as np
from numpy.linalg import det, inv
import scipy.stats as stats


class NaiveBayes:
    def __init__(self, train_features, train_targets, test_features, test_targets):
        """
        initialise input values
        """
        self.train_features = train_features
        self.train_targets = train_targets
        self.train_data = zip(train_features, train_targets)

        self.test_features = test_features
        self.test_targets = test_targets
        self.test_data = zip(test_features, test_targets)

    def _filter_data(self):
        """
        return input as a group of data
        """
        filtered_list = []
        target_set = set(self.train_targets)
        print 'filter - target_set: {0}'.format(target_set)
        zipped_data = zip(self.train_features, self.train_targets)

        # data for each target so to compute mu
        for target in target_set:
            filtered_data = filter(lambda row: row[1] == target, zipped_data)
            filtered_data = map(lambda grp: np.array(grp), zip(*filtered_data))
            filtered_list += [filtered_data]

        return filtered_list

    def params(self):
        """
        find mu for given target
        :param target: int
        """

        filtered_data = self._filter_data()
        means_list = []
        covars_list = []

        # data for each target so to compute mu
        for grp in range(len(filtered_data)):
            mean = np.mat(np.sum(filtered_data[grp][0], axis=0) /
                          len(filtered_data[grp][0]))
            means_list += [mean]

            covar_diff = np.mat(filtered_data[grp][0] - mean)
            covar = covar_diff.T * covar_diff / len(filtered_data[grp][0])
            covars_list += [covar]

        assert len(means_list) == len(covars_list)

        return means_list, covars_list

    def prob_x_y(self, pt, mean, covar):
        """
        Pr(x|y) - the probability distribution of a feature vector given label
        :return:
        """
        dim = self.train_features.shape[1]
        diff = pt - mean
        return 1 / np.sqrt((2*np.pi) ** dim * det(covar)) \
               * np.exp(-0.5 * diff * inv(covar) * diff.T)

    def prob_y_x(self, prior, pt, label_pos, means, covars):
        # TODO: redo
        """
        :param prior:
        :param pt:
        :param label_pos: int. position of the label concerned
        :param means:
        :param covars:
        :return:
        """
        sum_prob = 0
        likelihood = None

        for pos in range(len(means)):
            prob = self.prob_x_y(pt, means[pos], covars[pos])
            print prob * prior

            sum_prob += prob * prior

            if pos == label_pos:
                likelihood = prob * prior

        assert likelihood is not None

        print sum_prob
        return likelihood / sum_prob


class DiscLR:
    def __init__(self, features, targets):
        self.features = np.mat(features)
        self.targets = targets

        self.num_pts = len(self.targets)
        self.order = [[(0, 0)],
                      [(1, 1)],
                      [(2, 1)],
                      [(1, 1), (2, 1)],
                      [(1, 1), (2, 2)],
                      [(1, 2), (2, 1)],
                      [(1, 2), (2, 2)]]  # up to a degree of 5

    def _sig(self, phi, pt):
        """
        calculate the activation function
        :param phi_mat:
        :param pt:
        :return:
        """
        out = 1 / (1 + self._a(phi, pt))
        return out

    def _a(self, phi, pt):
        """
        calculate the exponential part of sig[a]
        :param phi:
        :param pt:
        :return:
        """
        phi_mat = np.mat(phi).T

        return np.exp(-phi_mat.T * pt)

    def trans_pt(self, pt, mode, degree):
        """
        transform a point into a column vector, or polynomially to the given degree
        :param pt:
        :param mode: None=point, 1=polynomial
        :return:
        """
        out = np.array([])

        if mode == 1:
            assert degree is not None

            pt = np.array(pt)[0]

            # print '\n---------------------------------------------'
            # print pt

            # grab info given degree - we need degree + 1 variables
            for d in range(degree+1):
                # print 'degree:', d
                temp_out = []

                for info in self.order[d]:
                    # print 'info:', info
                    temp_out += [np.power(pt[info[0]], info[1])]

                prod = np.product(temp_out)
                # print 'prod:', prod
                out = np.append(out, prod)
                # print ''

            z = np.mat(out).T
            # print 'z:\n', z
            # print '----------------------------------------------\n'
            return z

        elif not mode:
            return np.mat(pt).T

        else:
            assert False

    def ml_probability(self, phi, mode, degree):
        """
        Bernoulli distribution
        :param phi: np.array. row vector containing phi_0 and phi_1
        :param mode: None=point, 1=polynomial
        :param degree: int. Degree of polynomial required
        :return: the sum of probabilities
        """

        grp_0 = [self.targets[pt] *
                 np.log(self._sig(phi, self.trans_pt(self.features[pt], mode, degree)))
                 for pt in range(self.num_pts)]

        grp_1 = [(1 - self.targets[pt]) *
                 np.log(self._a(phi, self.trans_pt(self.features[pt], mode, degree)) /
                 (1 + self._a(phi, self.trans_pt(self.features[pt], mode, degree))))
                 for pt in range(self.num_pts)]

        final_sum = np.sum(grp_0, axis=0) + np.sum(grp_1, axis=0)
        np.ndarray.flatten(final_sum)
        print 'ml_prob: {0}'.format(-final_sum)
        return -final_sum

    def pt_probability(self, phi, pt):
        """
        calculate the probability of w given phi and a point
        :param phi:
        :param pt:
        :return:
        """
        # pt_arr = np.insert(np.array(pt), 0, 1.)
        return self._sig(phi, pt)

    def ml_gradient(self, phi, mode, degree):
        """
        linear ml_gradient
        :param phi:
        :param mode:
        :param degree:
        :return:
        """
        # non-linear (polynomial)
        if mode == 1:
            print 'non-linear'
            assert degree is not None

            grad = [self.trans_pt(self.features[pt], mode, degree) *
                    (self.targets[pt] -
                     self._sig(phi, self.trans_pt(self.features[pt], mode, degree)))
                    for pt in range(self.num_pts)]

        # linear
        elif mode is None:
            print 'linear'
            grad = [np.mat(self.features[pt]).T *
                    (self._sig(phi, np.mat(self.features[pt]).T) -
                     self.targets[pt])
                    for pt in range(self.num_pts)]

        else:
            assert False

        grad = np.sum(grad, axis=0)
        grad = np.ndarray.flatten(grad)
        print 'ml_gradient: {0}'.format(grad)

        if mode == 1:
            return -grad

        if mode is None:
            return grad

    def ml_hessian(self, phi, mode, degree):
        """
        Hessian function of the Bernoulli distribution
        :param phi: np.array. row vector containing phi_0 and phi_1
        :return: Hessian matrix (D+1 * D+1)
        """

        def f_sig(point):
            """
            sigmoid wrapper
            :param point: np.array. row feature vector
            """
            if mode is None:
                point = np.mat(point).T

            return self._sig(phi, point)

        if mode == 1:
            assert degree is not None
            hessian = np.sum([self.trans_pt(self.features[pt], mode, degree) *
                              f_sig(self.trans_pt(self.features[pt], mode, degree)) *
                              (1 - f_sig(self.trans_pt(self.features[pt], mode, degree))) *
                              self.trans_pt(self.features[pt], mode, degree).T
                              for pt in range(self.num_pts)], axis=0)

        elif mode is None:
            hessian = np.sum([np.mat(self.features[pt]).T *
                              f_sig(self.features[pt]) *
                              (1 - f_sig(self.features[pt])) *
                              np.mat(self.features[pt])
                              for pt in range(self.num_pts)], axis=0)
        else:
            assert False

        print 'hessian: {0}'.format(hessian)
        return hessian


class BayesLR(DiscLR):
    """
    extending from class DiscLR -
    """
    def __init__(self, features, targets, sigma_p):
        DiscLR.__init__(self, features, targets)
        self.sigma_p = sigma_p

        self.dim = self.features[0].shape[1]

    def _norm_dist(self, mean, covar, data, dim):
        """
        Normal distribution
        :param mean: the mean parameter
        :param covar: the covariance parameter
        :param dim: dimension of each feature vector
        :return:
        """
        diff = data - mean

        norm_dist = 1 / np.sqrt(((2*np.pi) ** dim) * det(covar)) * \
                    np.exp(-0.5 * diff.T * inv(covar) * diff)

        return norm_dist

    def _prior(self, phi):
        """
        Gaussian prior for Bayesian logistic regression
        """

        phi_mat = np.mat(phi).T

        # calculate covariance matrix
        covar = self.sigma_p * np.identity(self.dim)

        prior = self._norm_dist(0, covar, phi_mat, self.dim)
        print 'prior: {0}'.format(prior)

        # calculate normal distribution
        return prior

    def map_probability(self, phi):
        """
        probability taken prior into account
        :param sigma_p: prior
        """
        prob_0 = -self.ml_probability(phi, None, None)
        prob_1 = np.log(self._prior(phi))  # * self.num_pts

        prob = prob_0 + prob_1
        print 'map_prob: {0} || prob_0: {1}, prob_1: {2}'.format(-prob, prob_0, prob_1)

        return -prob

    def map_gradient(self, phi):
        phi_mat = np.mat(phi).T

        grad = np.mat(self.ml_gradient(phi, None, None)).T + (phi_mat / self.sigma_p)
        grad = np.ndarray.flatten(np.array(grad))
        print 'map_grad: {0}\n'.format(grad)

        return grad

    def map_hessian(self, phi):
        hessian = self.ml_hessian(phi, None, None) + 1 / self.sigma_p
        print 'map_hessian: {0}\n'.format(hessian)

        return hessian

    def laplace_approx(self, approx_phi):
        """
        Calculate the Laplace approximation using MAP optimised phi's
        """

        approx_phi_mat = np.mat(approx_phi).T

        mean = np.mat(approx_phi).T
        covar = -inv(self.map_hessian(approx_phi))
        print 'mean:\n{0}\n\ncovar:\n{1}'.format(mean, covar)

        approx_posterior = self._norm_dist(mean, covar, approx_phi_mat, self.dim)
        print 'map approx posterior: {0}'.format(approx_posterior)

        return approx_posterior
