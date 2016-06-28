import pandas as pd
import numpy as np
from datetime import datetime
import cPickle as pk

def convert_to_datetime(x):
    return datetime.fromtimestamp(x)

#preprocess data
def load_aggregate_data(filepath, house, channel):
    filename = filepath + '/' + house + '/' + channel + '.dat'
    agg_df = pd.read_table(filename, sep=' ',header= None,names = ['unix_date',channel])
    agg_df['date'] = agg_df['unix_date'].map(convert_to_datetime)
    agg_df = agg_df.set_index('date').drop('unix_date', axis = 1)
    return agg_df

# resample data at desired frequency interval and pivot data
def resample_and_pivot(df,resample_freq):
    avg_df = df.resample(resample_freq).mean()
    avg_df['time'] = avg_df.index.time
    avg_df['day'] = avg_df.index.date
    agg_pivot = avg_df.reset_index().pivot('time','day','meter_reading')
    return avg_df, agg_pivot

class DataStore(object):
    '''
    class to store measured power by channel and predictions when fitting the model
    '''
    def __init__(self, s3_bucket, house):
        self.house = house
        self.channels = {}
        self.labels = None
        self.url = 'http://s3.amazonaws.com/' + s3_bucket
        self.predictions = pd.DataFrame()

    def create_store(self, select_channels = None):
        """
        create datastore for selected channels
        :param select_channels: list of channel ID's to select, if none select all channels in the house
        :return: self
        """
        self.create_labels()
        self.create_channels(select_channels)


    def create_labels(self):
        filename = self.url + '/' + self.house + '/labels.dat'
        self.labels = pd.read_table(filename, sep=' ',header=None,names = ['channel_id','channel_desc'])

    def create_channels(self, select_channels = None):
        '''
        creates dictionary of pandas dataframes
        :param select_channels: list of channel id's to select, if None - select all channels in the house
        :return: populate self.channels dictionary
        '''
        if not select_channels:
            select_list = self.labels.channel_id.values
        else:
            select_list = select_channels
        for id in select_list:
            name = 'channel_' + str(id)
            print "Creating data frame for {}".format(name)
            self.channels[id] = load_aggregate_data(self.url, self.house, name)
            print "Done"

    def create_combined_df(self, start, end, freq=None, select_channels = None):
        '''
        resample dataframe into longer time periods and combine multiple channels into 1 dataframe
        :param freq: resampling frequency
        :param start: start date of observations
        :param end: end date of observations
        :param select_channels: list of channel ID's
        :return: aggregated dataframe
        '''
        if not select_channels:
            select_list = self.channels.keys()
        else:
            select_list = select_channels
        agg_df = self.select_window(select_list.pop(), start, end, resample_freq = freq)
        while select_list:
            ch_df = self.select_window(select_list.pop(), start, end, resample_freq = freq)
            agg_df = agg_df.join(ch_df, how = 'left')
        return agg_df

    def select_window(self, channel_id, start, end, resample_freq = None):
        '''
        Select subset of observations by specified dates
        :param channel_id: integer channel ID
        :param start: start date of observations
        :param end: end date of observations
        :param resample_freq: resample frequency
        :return: new dataframe
        '''
        if resample_freq:
            return self.channels[channel_id][start:end].resample(resample_freq).mean()
        return self.channels[channel_id][start:end]

    def select_top_k(self, k, period_start, period_end):
        totals = []
        channels = self.channels.keys()
        for channel in channels:
            total_power = self.channels[channel][period_start:period_end].sum().values[0]
            totals.append(total_power)

        idx = np.argsort(np.array(totals).flatten()).tolist()[::-1][:k]
        return [channels[i] for i in idx]


    def pickle_store(self, dest):
        with open(dest,'w') as f:
            pk.dump(self,f)


