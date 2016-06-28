import pandas as pd
import numpy as np
from datetime import datetime
import cPickle as pk
import json
from DataStore import DataStore
from FHMM import FHMM
from HMM import HMM, HMM_MAD
from Preprocessing import Appliance, train_test_split, create_matrix

with open('aws_keys.json') as f:
    data = json.load(f)
    ACCESS_KEY = data['access-key']
    SECRET_ACCESS_KEY = data['secret-access-key']



DStore = DataStore('ngalvbucket1', 'house_1')
DStore.create_store([1, 12, 5, 3, 10, 6, 9, 43, 7, 8])
top_10 = DStore.select_top_k(10,'2013-08-01','2013-09-01')
combined = DStore.create_combined_df('2013-06-01 00:00:00', '2013-10-31 23:59:59', select_channels = [12, 5, 3, 10, 6, 9, 43, 8], freq='1Min')
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
    hmm.fit_HMM()
    ModelDict[app.name] = hmm.model
    num_states_dict[app.name] = hmm.n_states

fhmm = FHMM()
fhmm.train(app_train_list,num_states_dict = num_states_dict)
predictions = pd.DataFrame()
DStore.predictions = fhmm.disaggregate(test_set2[['total']], DStore.predictions)

total_power_predicted = DStore.predictions.sum()
total_power_act = test_set2[predictions.columns].sum()

print total_power_predicted
print total_power_act

print sorted(predictions.columns)
print total_power_predicted.sort_index().values/total_power_act.sort_index().values

# with open('All_channels_1min.pkl', 'w') as f:
#     pk.dump(combined,f)


