from math import sqrt

import numpy as np
import pandas as pd
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmedoids import kmedoids

from utility.writer import save_clusters


def consensus_clustering(input_path, CLUSTERING_PATH):
    df = pd.read_csv(CLUSTERING_PATH + "distance_matrix.csv", delimiter=',', header=None)

    # read all the values
    sample = df.values
    # number of elements
    N = len(df.columns)
    print(isinstance(np.asmatrix(sample), np.matrix))
    # rule of thumbs for k
    df1 = pd.DataFrame(columns=['value'], index=['k_sqrtNBy4', 'k_sqrtNDiv4', 'k_sqrtNDiv2', 'k_sqrtNBy2', 'k_sqrtN', ])
    df1.at['k_1', 'value'] = 1
    df1.at['k_sqrtN', 'value'] = round(sqrt(N), 0)
    df1.at['k_sqrtNDiv2', 'value'] = round(sqrt(N / 2), 0)
    df1.at['k_sqrtNBy2', 'value'] = round(sqrt(N * 2), 0)
    df1.at['k_sqrtNDiv4', 'value'] = round(sqrt(N / 4), 0)
    df1.at['k_sqrtNBy4', 'value'] = round(sqrt(N * 4), 0)

    # Declare the weight of each vote
    # consensus matrix is NxN
    # initialization
    iterations = 20
    weight1 = 1 / iterations
    weight2 = 1 / len(df1.index)  # the amount of  k values used
    consensus_matrix = np.zeros((N, N))

    for k in df1.index:
        # run the same algorithm using several k values. Each configuration is run #iterations times.
        for iteration in range(iterations):
            k_value = int(df1.loc[k].values[0])
            initial_medoids = kmeans_plusplus_initializer(sample, k_value).initialize(return_index=True)
            kmedoids_instance = kmedoids(np.asmatrix(sample), initial_medoids, data_type="distance_matrix")
            kmedoids_instance.process()
            clusters = kmedoids_instance.get_clusters()
            coassociations_matrix = np.zeros((N, N))
            for cluster in clusters:
                for crypto in cluster:
                    # set the diagonal elements with value 1
                    coassociations_matrix[crypto][crypto] = 1
                    for crypto1 in cluster:
                        coassociations_matrix[crypto][crypto1] = 1
                        coassociations_matrix[crypto1][crypto] = 1
            # sum the two matrices
            consensus_matrix = consensus_matrix + coassociations_matrix
    consensus_matrix = consensus_matrix * weight1 * weight2
    # now, by doing (1 - consensus_matrix) we get the dissimilarity/distance matrix
    distance_matrix = 1 - consensus_matrix
    df = pd.DataFrame(data=distance_matrix)
    df.to_csv(CLUSTERING_PATH + "consensus_matrix(distance).csv", sep=",")

    # Hierarchical clustering
    for k in df1.index:
        k_value = int(df1.loc[k].values[0])

        agglomerative_instance = agglomerative(distance_matrix, k_value, type_link.AVERAGE_LINK)
        agglomerative_instance.process()
        # Obtain results of clustering
        clusters = agglomerative_instance.get_clusters()
        save_clusters(input_path, clusters, k, CLUSTERING_PATH)
