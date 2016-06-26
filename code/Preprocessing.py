import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import itertools

class Appliance():
    def __init__(self, name, power_data,cluster_means):
        self.name =  name
        self.power_data = power_data
        self.cluster_means = cluster_means


def cluster(x_train,x_test, max_number_clusters):
    """
    Iteratevely finds an optimal number of clusters based on silhouette score
    :param data: N*K numpy array, in case of a 1D array supply a column vector N*1
    :param max_number_clusters: integer, highest number of clusters
    :return: cluster centers
    """
    highest_score = -1
    for i in xrange(2,max_number_clusters):
        print "Fitting a KMeans model with {} clusters".format(i)
        kmeans = KMeans(n_clusters = i, n_jobs = -1).fit(x_train)
        labels = kmeans.predict(x_test)
        print "Calculating silhouette score..."
        s_score = silhouette_score(x_test, labels, sample_size = 10000)
        if s_score > highest_score:
            highest_score = s_score
            centers = kmeans.cluster_centers_
        print "Silhouette score with {} clusters:{}".format(i,s_score)
    print "Highest silhouete score of {} achieved with {} clusters\n".format(highest_score,len(centers))
    return centers

def Create_combined_states(df):
    columns = df.columns
    column_combinations = []
    for i in xrange(2,len(columns)+1):
        column_combinations = column_combinations + list(itertools.combinations(columns,i))

    for x in column_combinations:
        name = " ".join(list(x))
        df[name] = df[list(x)].sum(axis = 1)
    return df
