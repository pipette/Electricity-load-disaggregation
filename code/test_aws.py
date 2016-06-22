import pandas as pd
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pk
import boto
import os

with open('~/aws_keys.json') as f:
    data = json.load(f)
    ACCESS_KEY = data['access-key']
    SECRET_ACCESS_KEY = data['secret-access-key']

file2_url = 'http://s3.amazonaws.com/' + 'ngalvbucket1' + '/cancer_rates.csv'

df_test = pd.read_csv(file2_url)

with open('df_pickle.pkl', 'w') as f:
        pk.dump(df_test, f)
