'''
Implementation of the PAM algorithm for clustering
'''

import numpy as np
from scipy.spatial.distance import cdist
import random as rnd
from copy import deepcopy
import logging
from sklearn.datasets import make_blobs


def PAM_algorithm(data: np.ndarray,
                  k: int,
                  max_iter: int) -> tuple:
    '''
    Implementation of the PAM algorithm for clustering
    data: numpy array of shape (n, d)
    k: number of clusters
    max_iter: maximum number of iterations
    return: cluster assignments, cluster centers
    '''

    if type(data) is not np.ndarray:
        raise TypeError('data must be a numpy array')
    
    if type(k) is not int:
        raise TypeError('k must be an integer')
    
    if type(max_iter) is not int:
        raise TypeError('max_iter must be an integer')
    
    if k < 1:
        raise ValueError('Number of clusters must be greater than 0')
    
    if max_iter < 1:
        raise ValueError('Number of iterations must be greater than 0')

    n = data.shape[0]

    if k > n:
        raise ValueError('Number of clusters must be less than or equal to number')
    
    if k == n:
        return data, np.arange(n)

    # Initialize medoids
    medoids = rnd.sample(range(n), k)

    # Initialize cluster assignments
    cluster_assignments = np.zeros(n, dtype=int)

    # Initialize cluster centers
    cluster_centers = data[medoids]

    # Initialize distance matrix
    dist_matrix = cdist(data, cluster_centers, metric='cityblock')

    # Assign clusters
    cluster_assignments = np.argmin(dist_matrix, axis=1)

    for i in range(max_iter):
        # Copy previous medoids
        prev_medoids = deepcopy(medoids)

        # Update medoids
        for j in range(k):
            cluster_j = np.where(cluster_assignments == j)[0]
            dist_matrix[cluster_j, j] = cdist(data[cluster_j], data[cluster_j], metric='cityblock').sum(axis=1)
            medoids[j] = cluster_j[np.argmin(dist_matrix[cluster_j, j])]

        # Update cluster centers
        cluster_centers = data[medoids]

        # Update distance matrix
        dist_matrix = cdist(data, cluster_centers, metric='cityblock')

        # Assign clusters
        cluster_assignments = np.argmin(dist_matrix, axis=1)

        # Check for convergence
        if np.array_equal(prev_medoids, medoids):
            break

    return cluster_centers, cluster_assignments


def cluster_PAM(data: np.ndarray,
                k: int,
                max_iter: int = 1000,
                n_of_runs: int = 10) -> tuple:
    '''
    Cluster data using the PAM algorithm
    data: numpy array of shape (n, d)
    k: number of clusters
    max_iter: maximum number of iterations in single run
    n_of_runs: number of runs to perform
    return: cluster centers, cluster assignments
    '''
    best_cost = np.inf

    for _ in range(n_of_runs):
        cluster_centers, cluster_assignments = PAM_algorithm(data, k, max_iter)
        cost = np.sum(cdist(data, cluster_centers, metric='cityblock')[np.arange(data.shape[0]), cluster_assignments])
        if cost < best_cost:
            best_cost = cost
            best_cluster_centers = cluster_centers
            best_cluster_assignments = cluster_assignments

    return best_cluster_centers, best_cluster_assignments


def compare_assignments(true_assignments, cluster_assignments) -> bool:
    '''
    Compare true assignments with cluster assignments
    true_assignments: true cluster assignments
    cluster_assignments: cluster assignments
    return: True if the assignments are the same, False otherwise
    '''
    if len(true_assignments) != len(cluster_assignments):
        raise ValueError('Length of true_assignments and cluster_assignments must be the same')
    
    if len(set(true_assignments)) != len(set(cluster_assignments)):
        return ValueError('Number of clusters in true_assignments and cluster_assignments must be the same')
    
    n = len(true_assignments)
    cluster_names = {}
    for i in range(n):
        if cluster_assignments[i] in cluster_names:
            if true_assignments[i] != cluster_names[cluster_assignments[i]]:
                return False
            else:
                continue
        else:
            cluster_names[cluster_assignments[i]] = true_assignments[i]
            continue          
    return True


def main():
    rnd.seed(42)

    logging.basicConfig(level=logging.INFO)
    logging.info('Testing PAM algorithm')

    # ----- Test 1 -----
    data1 = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    class_labels1 = np.array([0, 0, 0, 1, 1, 1])
    k1 = 2

    logging.info(f'Data in test 1: {data1}')
    logging.info(f'Class labels in test 1: {class_labels1}')
    cluster_centers1, cluster_assignments1 = cluster_PAM(data1, k1)

    logging.info(f'Cluster assignments: {cluster_assignments1}')
    if compare_assignments(cluster_assignments1, class_labels1):
        logging.info('Test 1 passed')
    else:
        logging.error('Test 1 failed')
    
    # ----- Test 2 -----
    data2, class_labels2 = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    k2 = 3

    logging.info(f'Data in test 2: {data2}')
    logging.info(f'Class labels in test 2: {class_labels2}')
    cluster_centers2, cluster_assignments2 = cluster_PAM(data2, k2)

    logging.info(f'Cluster assignments: {cluster_assignments2}')
    if compare_assignments(cluster_assignments2, class_labels2):
        logging.info('Test 2 passed')
    else:
        logging.error('Test 2 failed')

    # ----- Test 3 -----
    data3, class_labels3 = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)
    k3 = 5

    logging.info(f'Data in test 3: {data3}')
    logging.info(f'Class labels in test 3: {class_labels3}')
    cluster_centers3, cluster_assignments3 = cluster_PAM(data3, k3)

    logging.info(f'Cluster assignments: {cluster_assignments3}')
    if compare_assignments(cluster_assignments3, class_labels3):
        logging.info('Test 3 passed')
    else:
        logging.error('Test 3 failed')
        import matplotlib.pyplot as plt

        plt.scatter(data3[:, 0], data3[:, 1], c=cluster_assignments3)
        plt.title('Test 3 - predicted cluster assignments')
        plt.show()

        plt.scatter(data3[:, 0], data3[:, 1], c=class_labels3)
        plt.title('Test 3 - true cluster assignments')
        plt.show()
        '''
        Test fails, but looking at the plots, the clustering is accurate
        It is wrong in just one point, which is an outlier
        '''


if __name__ == '__main__':
    main()
