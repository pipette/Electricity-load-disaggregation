import pandas as pd
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

file2_url = 'http://s3.amazonaws.com/' + 'ngalvbucket1' + '/house_1'

def convert_to_datetime(x):
    return datetime.fromtimestamp(x)

#preprocess data
def load_aggregate_data(house, channel):
    filename = '../data/' + house + '/' + channel + '.dat'
    agg_df = pd.read_table(filename, sep=' ',header=None,names = ['unix_date','meter_reading'])
    agg_df['date'] = agg_df['unix_date'].map(convert_to_datetime)
    agg_df = agg_df.set_index('date').drop('unix_date', axis = 1)
    agg_df['time'] = agg_df.index.time
    agg_df['day'] = agg_df.index.date
    return agg_df

# resample data at desired frequency interval and pivot data
def resameple_and_pivot(df,resample_freq):
    avg_df = df.resample(resample_freq).mean()
    avg_df['time'] = avg_df.index.time
    avg_df['day'] = avg_df.index.date
    agg_pivot = avg_df.reset_index().pivot('time','day','meter_reading')
    return avg_df, agg_pivot


house = 'house_1'
channel = 'labels'
filename = '../data/' + house + '/' + channel + '.dat'
h1_labels = pd.read_table(filename, sep=' ',header=None,names = ['unix_date','meter_reading'])


agg_data = load_aggregate_data('house_1', 'channel_1')
agg_data = agg_data['2014-03-01':'2014-04-01'].rename(columns = {'meter_reading':'aggregate'})
agg_data = agg_data.resample('1Min').mean()

for row in h1_labels.values[1:]:
    single_df = load_aggregate_data('house_1', 'channel_'+str(row[0]))
    single_df = single_df['2014-03-01':'2014-05-01'][['meter_reading']].rename(columns = {'meter_reading':row[1]})
    single_df = single_df.resample('1Min').mean()
    agg_data = agg_data.join(single_df, how = 'left')

print agg_data.head()

with open('df_pickle.pkl', 'w') as f:
        pk.dump(agg_data, f)
