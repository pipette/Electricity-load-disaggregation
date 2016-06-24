import pandas as pd
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import cPickle as pk
from hmmlearn.hmm import GaussianHMM


class HMM():
    def __init__(self, X_test, X_train, n_states = 2):
        self.X_train = X_train
        self.X_test = X_test
        self.n_states = n_states
        self.model =  GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=10000)

    def fit_HMM(self):
        print "fitting to HMM and decoding ..."
        self.model.fit(self.X_train)
        print "done"

    def extract_means(self):
        return self.model.means_[:,0].flatten()

    def HMM_total_accuracy(self, obs_levels, state_means):
        hidden_states = self.model.predict(obs_levels)
        predict_levels = [state_means[state] for state in hidden_states]
        test_error = 1 - (np.sum(obs_levels[:,0]) - np.sum(predict_levels))/np.sum(obs_levels[:,0])
        return test_error

    def HMM_MAD_perc(self,obs_levels, state_means):
        hidden_states = self.model.predict(obs_levels)
        predict_levels = [state_means[state] for state in hidden_states]
        errors = np.abs(obs_levels[:,0] - predict_levels)/obs_levels[:,0]
        return np.mean(errors)

    def run(self):
        self.fit_HMM()
        state_means = self.extract_means()
        test_error = self.HMM_accuracy(self.X_test, state_means)
        print "Accuracy for the model with {} hidden states is: {}".format(self.n_states,test_error)

