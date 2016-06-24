import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster(data, max_number_clusters):
    """
    Iteratevely finds an optimal number of clusters based on silhouette score
    :param data: numpy array
    :param max_number_clusters: integer, highest number of clusters
    :return: cluster centers
    """
    highest_score = 0
    x = data.reshape((-1,1))
    for i in xrange(2,max_number_clusters):
        kmeans = KMeans(n_clusters = i).fit(x)
        labels = kmeans.predict(x)
        s_score = silhouette_score(x, labels)
        if s_score > highest_score:
            highest_score = s_score
            centers = kmeans.cluster_centers_
    return centers