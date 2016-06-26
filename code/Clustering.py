import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster(x, max_number_clusters):
    """
    Iteratevely finds an optimal number of clusters based on silhouette score
    :param data: N*K numpy array, in case of a 1D array supply a column vector N*1
    :param max_number_clusters: integer, highest number of clusters
    :return: cluster centers
    """
    highest_score = 0
    for i in xrange(2,max_number_clusters):
        kmeans = KMeans(n_clusters = i, n_jobs = -1).fit(x)
        labels = kmeans.predict(x)
        s_score = silhouette_score(x, labels)
        if s_score > highest_score:
            highest_score = s_score
            centers = kmeans.cluster_centers_
    return centers