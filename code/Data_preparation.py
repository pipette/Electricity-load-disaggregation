import pandas as pd
import numpy as np
from datetime import datetime
import cPickle as pk
import json
from DataStore import DataStore
# from FHMM import FHMM
# from HMM import HMM, HMM_MAD
from Preprocessing import Appliance, train_test_split, create_matrix

with open('aws_keys.json') as f:
    data = json.load(f)
    ACCESS_KEY = data['access-key']
    SECRET_ACCESS_KEY = data['secret-access-key']



DStore = DataStore('ngalvbucket1', 'house_1')
all_channels = [1, 12, 5, 3, 10, 6, 9, 43, 7, 8]
select_channels = [12, 5, 3, 10, 6, 9, 43, 8]
# select_channels = [12, 5, 6]

DStore.create_store(all_channels)
top_10 = DStore.select_top_k(10,'2013-08-01','2013-09-01')

combined = DStore.create_combined_df('2013-06-01 00:00:00', '2013-10-31 23:59:59', select_channels = select_channels, freq='1Min')

with open('combined.pkl', 'w') as f:
    pk.dump(combined,f)



