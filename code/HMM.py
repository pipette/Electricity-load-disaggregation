import pandas as pd
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import cPickle as pk
from hmmlearn.hmm import GaussianHMM
import math

def HMM_MAD(model,obs_levels):
    hidden_states = model.predict(obs_levels)
    means = model.means_.round().astype(int).flatten().tolist()
    predict_levels = np.array([means[state] for state in hidden_states]).reshape(obs_levels.shape)
    abs_error = np.absolute(obs_levels - predict_levels)
    return np.mean(abs_error)/np.mean(obs_levels)

def perc_std_expl(model,obs_levels):
    """
    :param observed: df of observed energy levels per channel
    :param predicted: df of predicted energy levels per channel
    :return: percentage of standard deviation explained
    """
    hidden_states = model.predict(obs_levels)
    means = model.means_.round().astype(int).flatten().tolist()
    predict_levels = np.array([means[state] for state in hidden_states]).reshape(obs_levels.shape)
    obs_mean = np.mean(obs_levels)
    r2 = 1 - (np.sum((obs_levels - predict_levels)**2))/np.sum((obs_levels - obs_mean)**2)
    return 1 - math.sqrt(1-r2)

def r2(model,obs_levels):
    hidden_states = model.predict(obs_levels)
    means = model.means_.round().astype(int).flatten().tolist()
    predict_levels = np.array([means[state] for state in hidden_states]).reshape(obs_levels.shape)
    obs_mean = np.mean(obs_levels)
    r2 = 1 - (np.sum((obs_levels - predict_levels)**2))/np.sum((obs_levels - obs_mean)**2)
    return r2

class HMM():
    def __init__(self, X_test, X_train):
        self.X_train = X_train
        self.X_test = X_test
        self.n_states = None
        self.model =  None

    def fit_HMM(self,error_metric):
        print "Looking for optimal number of states and fitting HMM"
        for i in xrange(2,5):
            candidate = GaussianHMM(n_components=i, covariance_type="full", n_iter=1000)
            candidate.fit(self.X_train)
            if error_metric == HMM_MAD:
                error = HMM_MAD(candidate,self.X_test)
                if i == 2:
                    best_guess = error
                    best_model = candidate
                    opt_n_states = i
                else:
                    if error < best_guess:
                        opt_n_states = i
                        best_model = candidate
                        best_guess = error
            else:
                error = error_metric(candidate,self.X_test)
                if i == 2:
                    best_guess = error
                    best_model = candidate
                    opt_n_states = i
                else:
                    if error > best_guess:
                        opt_n_states = i
                        best_model = candidate
                        best_guess = error
        self.model = best_model
        self.n_states = opt_n_states
        print "Done. Lowest error of {} achieved with {} states".format(best_guess, opt_n_states)

    def extract_means(self):
        return self.model.means_[:,0].flatten()

    def HMM_total_accuracy(self, obs_levels, state_means):
        hidden_states = self.model.predict(obs_levels)
        predict_levels = [state_means[state] for state in hidden_states]
        test_error = 1 - (np.sum(obs_levels[:,0]) - np.sum(predict_levels))/np.sum(obs_levels[:,0])
        return test_error

    def HMM_MAD_perc(self,obs_levels, state_means):
        hidden_states = self.model.predict(obs_levels)
        predict_levels = np.array([state_means[state] for state in hidden_states]).reshape(obs_levels.shape)
        abs_error = np.absolute(obs_levels - predict_levels)
        return np.mean(abs_error)/np.mean(obs_levels)

    def run(self):
        self.fit_HMM()
        state_means = self.extract_means()
        test_error = self.HMM_accuracy(self.X_test, state_means)
        print "Accuracy for the model with {} hidden states is: {}".format(self.n_states,test_error)

