import pandas as pd
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import cPickle as pk
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/4app_train.pkl') as f:
    four_app_train = pk.load(f)
with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/4app_test.pkl') as f:
    four_app_test = pk.load(f)

four_app_train.fillna(value = 0,inplace = True)
four_app_test.fillna(value = 0,inplace = True)

def HMM_accuracy(obs_levels,hidden_states,state_means):
    predict_levels = [state_means[state] for state in hidden_states]
    test_error = 1 - (np.sum(obs_levels) - np.sum(predict_levels))/np.sum(obs_levels)
    return test_error

def fit_HMM(X_train,X_test, n_states):
    print "fitting to HMM and decoding ..."

    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=10000).fit(X_train)
    state_means = model.means_.flatten()
    print state_means
    # Predict the optimal sequence of internal hidden state
    hidden_states_train = model.predict(X_train)
    hidden_states_test = model.predict(X_test)




class HMM():

    def __init__(self, X_test, X_train, n_states = 2):
        self.X_train = X_train
        self.X_test = X_test
        self.n_states = n_states
        self.model = None

    def fit_HMM(self):
        print "fitting to HMM and decoding ..."
        self.model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=10000).fit(self.X_train)
        print "done"

    def extract_means(self):
        return self.model_means.flatten()

    def HMM_accuracy(self, obs_levels, state_means):
        hidden_states = self.model.predict(obs_levels)
        predict_levels = [state_means[state] for state in hidden_states]
        test_error = 1 - (np.sum(obs_levels) - np.sum(predict_levels))/np.sum(obs_levels)
        return test_error