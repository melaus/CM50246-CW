from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from task3 import NaiveBayes as nb, DiscLR as LR, BayesLR as BLR
import generate_classification_data as gen
from scipy.optimize import minimize
import scipy.stats as stats

# iris = datasets.load_iris()
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(np.array([iris.data[:,1]]).T, iris.target).predict(np.array([iris.data[:,1]]).T)
# print y_pred
# # print("Number of mislabeled points out of a total %d points : %d"
        # # % (np.array([iris.data[:,1]]).T.shape[0],(iris.target != y_pred).sum()))


### Naive Bayes ###
########################
# num_pts = 500
# num_train_pts = 250
# num_test_pts = 250
# num_targets=2
# gen_features, gen_targets = gen.generate_classification_data(test_case=0,
#                                                              num_training_points=num_pts,
#                                                              plot_data=False)
# test_features = gen_features[np.r_[125:250, 375:500]]
# test_targets = gen_targets[np.r_[125:250, 375:500]]
#
# train_features = gen_features[np.r_[0:125, 250:375]]
# train_targets = gen_targets[np.r_[0:125, 250:375]]
#
# print 'split feature shapes - train: {0} | test: {1}'.format(train_features.shape, test_features.shape)
# print 'split target shapes - train: {0} | test: {1}'.format(train_targets.shape, test_targets.shape)
#
# # means and covariances
# n = nb(train_features, test_targets, test_features, test_targets)
# means, covars = n.params()
# print 'means: {0}'.format(means)
# print 'covariances: {0}'.format(covars)
#
# # likelihoods
#
# likelihoods = np.array([n.prob_x_y(test_features[pt_pos], means[target_pos], covars[target_pos])
#                         for target_pos in range(num_targets)
#                         for pt_pos in range(num_test_pts)
#                        ]).reshape(num_targets, num_test_pts)
#
# posterior = likelihoods / np.sum(likelihoods, axis=0)
# y_pred = map(lambda pair: pair.index(max(pair)), zip(posterior[0], posterior[1]))
# print 'prediction:', y_pred
# print 'targets:', test_targets
# print 'differences:', sum(map(lambda pair: int(pair[0] != pair[1]), zip(test_targets, y_pred)))
##########################



### ML, MAP, non-linear ###
###########################
num_pts = 60

# datasets
gen_features, gen_targets = gen.generate_classification_data(test_case=0,
                                                             num_training_points=num_pts,
                                                             plot_data=False)

train_features = np.insert(gen_features, 0, 1, axis=1)
train_targets = gen_targets  # [np.r_[0:125, 250:375]]

gen_features_1, gen_targets_1 = gen.generate_classification_data(test_case=1,
                                                                 num_input_dimensions=2,
                                                                 plot_data=False)
train_features_1 = np.insert(gen_features_1, 0, 1, axis=1)
train_targets_1 = gen_targets_1

gen_features_2, gen_targets_2 = gen.generate_classification_data(test_case=2,
                                                                 num_input_dimensions=2,
                                                                 plot_data=False)
train_features_2 = np.insert(gen_features_2, 0, 1, axis=1)
train_targets_2 = gen_targets_2


lr =     LR(train_features_1, train_targets_1)
bayes = BLR(train_features_1, train_targets_1, 2)
# BFGS, L-BFGS-B, Newton-CG, TNC

# toggle between None and 1 for linear and non-linear
# phi needs degree + 1 values for non-linear
# non-linear only works with ML
mode = 1
degree = 5
phi = np.array([1,1,1,1,1,1])

print '-----------------opt_ml-----------------'
opt_ml = minimize(lr.ml_probability, phi,
                  jac=lr.ml_gradient,
                  method='TNC',
                  hess=lr.ml_hessian,
                  args=(mode, degree))
print opt_ml


# print '-----------------opt_map-----------------'
# opt_map = minimize(bayes.map_probability, phi,
#                    jac=bayes.map_gradient,
#                    method='TNC',
#                    hess=bayes.map_hessian)
# print opt_map
##########################

###############
# PLOT non-linear!
x_vals = np.arange(-1, 1, 0.04)
z = np.zeros([len(x_vals), len(x_vals)])

for x0 in range(len(x_vals)):
    for x1 in range(len(x_vals)):
        p = lr.trans_pt(np.mat([1., x_vals[x1], x_vals[x0]]), mode, degree)
        z[x0, x1] = lr.pt_probability(opt_ml.x, p)[0, 0]  # opt_map.x for MAP, opt_ml.x for ML

# z = np.array(z).reshape(len(x_vals), len(x_vals))

plt.scatter(gen_features_1[0:num_pts/2, 0], gen_features_1[0:num_pts/2, 1], c='g')
plt.scatter(gen_features_1[num_pts/2:,  0], gen_features_1[num_pts/2:,  1], c='r')
plt.imshow(z, cmap='hot', extent=[-1, 1, -1, 1])
plt.title('Test case 1 (non-linear, ML, TNC, degree=5, phi values=1)\n Pr(w = 1 | X)')
plt.savefig('test_1-non_tnc_norm.png', dpi=200)
# plt.show()
###############

