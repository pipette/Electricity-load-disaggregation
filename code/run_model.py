import cPickle as pk
import numpy as np
import FHMM




with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/4app_train.pkl') as f:
    four_app_train = pk.load(f)

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/4app_test.pkl') as f:
    four_app_test = pk.load(f)

class Appliance():
    def __init__(self, name, power_data):
        self.name =  name
        self.power_data = power_data

app_train_list = []
app_test_list = []
for channel in four_app_train[['channel_12','channel_7']].columns:
    app_train_list.append(Appliance(channel,four_app_train[[channel]]))
    app_test_list.append(Appliance(channel,four_app_test[[channel]]))

num_states_dict={}
for channel in four_app_train.columns:
    num_states_dict[channel]=2

test_mains = four_app_test[['channel_12','channel_7']]
test_mains['total'] = test_mains.sum(axis = 1)
# print test_mains.head(20)

fhmm = FHMM.FHMM()
fhmm.train(app_train_list, num_states_dict = num_states_dict)
predict= fhmm.disaggregate_chunk(test_mains[['total']])

for ind in fhmm.individual:
    print fhmm.individual[ind].transmat_
    print "*"*100

print fhmm.model.transmat_
# print predict.head(20)

