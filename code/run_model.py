import cPickle as pk
import pandas as pd
import numpy as np
from FHMM import FHMM
from HMM import HMM, HMM_MAD, perc_std_expl,r2
from Preprocessing import Create_combined_states, Appliance, train_test_split,create_matrix
import math

def perc_std_expl_full(pred_df,obs_df):
    """
    :param observed: df of observed energy levels per channel
    :param predicted: df of predicted energy levels per channel
    :return: percentage of standard deviation explained
    """
    return_dict = {}
    for channel in pred_df:
        X_pred = pred_df[[channel]].values
        X_obs = obs_df[[channel]].values
        obs_mean = np.mean(X_obs)
        r2 = 1 - (np.sum((X_obs - X_pred)**2))/np.sum((X_obs - obs_mean)**2)
        return_dict[channel] = 1 - math.sqrt(1-r2)
    return return_dict

def r2_full(pred_df,obs_df):
    return_dict = {}
    for channel in pred_df:
        X_pred = pred_df[[channel]].values
        X_obs = obs_df[[channel]].values
        obs_mean = np.mean(X_obs)
        r2 = 1 - (np.sum((X_obs - X_pred)**2))/np.sum((X_obs - obs_mean)**2)
        return_dict[channel] = r2
    return return_dict


with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/combined.pkl') as f:
    total = pk.load(f)

print total.head()
combined = total['2013-06-01 00:00:00': '2013-09-30 23:59:59'][['channel_12','channel_5','channel_6','channel_3']]

train_set, test_set1, test_set2 = train_test_split(combined,'2013-07-31 23:59:59','2013-08-31 23:59:59')

app_train_list = []
app_test_list1 = []
app_test_list2 = []

for channel in combined.columns:
    app_train_list.append(Appliance(channel,train_set[[channel]]))
    app_test_list1.append(Appliance(channel,test_set1[[channel]]))
    app_test_list2.append(Appliance(channel,test_set2[[channel]]))

num_states_dict={}
ModelDict = {}

for i,app in enumerate(app_train_list):
    X_train = create_matrix(app,good_chunks = True)
    X_test = create_matrix(app_test_list1[i],good_chunks = False)
    hmm = HMM(X_train,X_test)
    print app.name
    hmm.fit_HMM(perc_std_expl)
    ModelDict[app.name] = hmm.model
    num_states_dict[app.name] = hmm.n_states

fhmm = FHMM()
fhmm.train(app_train_list,num_states_dict = num_states_dict)
predictions = pd.DataFrame()
predictions = fhmm.disaggregate(test_set2[['total']], predictions)

total_power_predicted = predictions.sum()
total_power_act = test_set2[predictions.columns].sum()

print "Percent stand.dev.explained, 1 min:", perc_std_expl_full(predictions,test_set2)
print "R2, 1 min:" , r2_full(predictions,test_set2)

predictions_15Min = predictions.resample('15Min').sum()
test_15Min = test_set2.resample('15Min').sum()

print "Percent stand.dev.explained, 15 min:", perc_std_expl_full(predictions_15Min,test_15Min)
print "R2, 15 min:" , r2_full(predictions_15Min,test_15Min)

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/predictions.pkl','w') as f:
    pk.dump(predictions,f)

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/test2.pkl','w') as f:
    pk.dump(test_set2,f)

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/Trained_model.pkl','w') as f:
    pk.dump(fhmm,f)
