import argparse
from math import sqrt
from generate_unsupervised_data import generate_unsupervised_data
from random import sample
import numpy as np
import matplotlib.pyplot as plt


def generate_centroids(data, num_centroids):
    """
    generate centroids by randomly selecting from the required number of centroids from the data
    :param data: np.array (transposed). All the data points in (x,y) 'pairs'
    :param num_centroids: int. Command-line input to specify the required number of centroids
    :return:
    """
    return np.array(sample(data, num_centroids))


def kmeans(data, centroids):
    """
    initialise k-means with correct points
    :param data: list. A list of points.
    :param centroids: list. A list of centroid points
    :return:
    """
    cur_cen_list = centroids
    # #######
    # with open('centroids.txt', 'w') as f:
    #     f.write(str(centroids))
    # #######

    count = 1

    while True:
        # print 'iteration', count
        new_cen_list = []
        distances = [[[pt, cen, sqrt((cen[0]-pt[0])**2 + (cen[1]-pt[1])**2)] for cen in cur_cen_list] for pt in data]

        # ######
        # with open('data.txt', 'w') as f:
        #     f.write(str(distances))
        # ######

        pts_min_dis = [min(pt_info, key=lambda info: np.min(info[2])) for pt_info in distances]

        # calculate new centroids based on points around current centroids
        for cen in cur_cen_list:
            # print cen
            assert type(cen) == np.ndarray
            cen_pts_list = filter(lambda info: np.array_equal(info[1], cen), pts_min_dis)
            cen_x = map(lambda item: item[0][0], cen_pts_list)
            cen_y = map(lambda item: item[0][1], cen_pts_list)
            assert len(cen_x) == len(cen_y)
            new_cen_list += [(sum(cen_x)/len(cen_x), sum(cen_y)/len(cen_y))]

        new_cen_list = np.array(new_cen_list)  # make list into a numpy array

        if (np.sort(cur_cen_list, axis=0) == np.sort(new_cen_list, axis=0)).all():
            # end loop when the centroids don't change anymore - return transposed version for easy plotting
            print 'iterations: {0}'.format(count)
            return np.array(map(lambda info: info[0], pts_min_dis)).T, new_cen_list.T
        else:
            # assign new list as the current list
            cur_cen_list = new_cen_list
        count += 1


def plot_outcome(data, centroids, title='New Centroids'):
    plt.figure(2)
    plt.plot(data[0, :], data[1, :], 'kx')
    plt.plot(centroids[0, :], centroids[1, :], 'rx', mew=5, ms=10)
    plt.title(title)
    plt.show()


def main():
    args = get_args()
    print args
    pts, true_dist = generate_unsupervised_data(num_dimensions=args.d,
                                                num_clusters=args.k,
                                                num_points=args.p,
                                                plot_data=True)

    pts_min_dis, new_cen_list = kmeans(data=pts.T, centroids=generate_centroids(pts.T, args.c))
    plot_outcome(pts_min_dis, new_cen_list)


def get_args():
    """
    get CLi arguments
    :return: argparse
    """
    parser = argparse.ArgumentParser(description='K-means')
    parser.add_argument('--d', '-num_dimensions', type=int, default=2, help='the number of dimensions')
    parser.add_argument('--k', '-num_clusters', type=int, default=5, help='the number of clusters')
    parser.add_argument('--c', '-num_centroids', type=int, default=5, help='the number of clusters')
    parser.add_argument('--p', '-num_points', type=int, default=500, help='the number of clusters')

    return parser.parse_args()


if __name__ == '__main__':
    main()
