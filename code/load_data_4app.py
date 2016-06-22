import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pk


def convert_to_datetime(x):
    return datetime.fromtimestamp(x)

#preprocess data
def load_aggregate_data(house, channel):
    filename = '/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/' + house + '/' + channel + '.dat'
    agg_df = pd.read_table(filename, sep=' ',header=None,names = ['unix_date',channel])
    agg_df['date'] = agg_df['unix_date'].map(convert_to_datetime)
    agg_df = agg_df.set_index('date').drop('unix_date', axis = 1)
    return agg_df

# resample data at desired frequency interval and take first differences
def resample_df(df,resample_freq):
    avg_df = df.resample(resample_freq).mean()
    name = df.columns[0]
    avg_df[name+'_diff'] = avg_df[name] - avg_df[name].shift(1)
    return avg_df

#focus on fridge, tv, dishwasher and washer for now


# training data june and july 2013
#testing data august and september 2013

train_start = '2013-06-01 00:00:00'
train_end = '2013-07-01 23:59:59'

test_start = '2013-08-01 00:00:00'
test_end = '2013-09-01 23:59:59'
res_freq = '1Min'

fridge = load_aggregate_data('house_1', 'channel_12')
fridge_train = resample_df(fridge[train_start:train_end],res_freq)
fridge_test = resample_df(fridge[test_start:test_end],res_freq)

washer = load_aggregate_data('house_1', 'channel_5')
washer_train = resample_df(washer[train_start:train_end], res_freq)
washer_test = resample_df(washer[test_start:test_end], res_freq)

tv = load_aggregate_data('house_1','channel_7')
tv_train = resample_df(tv[train_start:train_end],res_freq)
tv_test = resample_df(tv[test_start:test_end],res_freq)

dishwasher = load_aggregate_data('house_1','channel_6')
dw_train = resample_df(dishwasher[train_start:train_end],res_freq)
dw_test = resample_df(dishwasher[test_start:test_end],res_freq)

four_app_train = fridge_train
four_app_test = fridge_test

count = 0

df_train = [fridge_train,washer_train,tv_train,dw_train]
df_test = [fridge_test,washer_test,tv_test,dw_test]

for idx in xrange(len(df_train)):
    with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/' +df_train[idx].columns[0]+ '_train.pkl','w') as f:
        pk.dump(df_train[idx],f)
    with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/' +df_test[idx].columns[0]+ '_test.pkl','w') as f:
        pk.dump(df_test[idx],f)

    if count != 0:
        four_app_train = four_app_train.join(df_train[idx], how = 'left')
        four_app_test = four_app_test.join(df_test[idx], how = 'left')
    count += 1


with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/4app_train.pkl','w') as f:
    pk.dump(four_app_train,f)

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/4app_test.pkl','w') as f:
    pk.dump(four_app_test,f)




