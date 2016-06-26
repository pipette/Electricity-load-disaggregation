import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cPickle as pk
import numpy as np


def cluster(x, max_number_clusters):
    """
    Iteratevely finds an optimal number of clusters based on silhouette score
    :param data: N*K numpy array, in case of a 1D array supply a column vector N*1
    :param max_number_clusters: integer, highest number of clusters
    :return: cluster centers
    """
    highest_score = -1
    for i in xrange(2,max_number_clusters):
        print "Fitting a KMeans model with {} clusters".format(i)
        kmeans = KMeans(n_clusters = i, n_jobs = -1).fit(x)
        labels = kmeans.predict(x)
        print "Calculating silhouette score..."
        s_score = silhouette_score(x, labels)
        if s_score > highest_score:
            highest_score = s_score
            centers = kmeans.cluster_centers_
        print "Silhouette score with {} clusters:{}".format(i,s_score)
    print "Highest silhouete score of {} achieved with {} clusters\n".format(highest_score,len(centers))
    return centers


class Appliance():
    def __init__(self, name, power_data,cluster_means):
        self.name =  name
        self.power_data = power_data
        self.cluster_means = cluster_means

with open('4app_train.pkl') as f:
    four_app_train = pk.load(f)

with open('4app_test.pkl') as f:
    four_app_test = pk.load(f)

app_train_list = []
app_test_list = []

for channel in four_app_train.columns[::2]:
    power_data = four_app_train[[channel]].fillna(value = 0,inplace = False)
    X = power_data.values.reshape((-1, 1))
    print "*"*100
    print "Looking for optimal number of clusters for {} \n".format(channel)
    X_centers = cluster(X,4)
    app_train_list.append(Appliance(channel,power_data,X_centers))
    app_test_list.append(Appliance(channel,four_app_test[[channel]],X_centers))

train_and_test_apps = [app_train_list,app_test_list]
with open('app_list.pkl','w') as f:
    pk.dump(train_and_test_apps,f)




