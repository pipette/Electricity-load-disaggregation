import pandas as pd
import cPickle as pk
import itertools
from Preprocessing import Create_combined_states

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/4app_train.pkl') as f:
    four_app_train = pk.load(f)

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/4app_test.pkl') as f:
    four_app_test = pk.load(f)

columns = four_app_train.columns[::2]
four_app_train = four_app_train[columns]
four_app_test = four_app_test[columns]
new_df = Create_combined_states(four_app_train)

print new_df.head()



