import cPickle as pk
import pandas as pd
import numpy as np
import FHMM
from HMM import HMM, HMM_MAD
from Preprocessing import Create_combined_states, Appliance



with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/4app_train.pkl') as f:
    four_app_train = pk.load(f)

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/4app_test.pkl') as f:
    four_app_test = pk.load(f)



columns = ['channel_5','channel_6','channel_12']
    # four_app_train.columns[::2]
four_app_train = four_app_train[columns]
four_app_test = four_app_test[columns]
four_app_test['total'] = four_app_test[columns].sum(axis = 1)
new_df_train = Create_combined_states(four_app_train)
new_df_test = Create_combined_states(four_app_test)


app_train_list = []
app_test_list1 = []


# for channel in four_app_train.columns:
#     app_train_list.append(Appliance(channel,new_df_train[[channel]],[0,0]))
#     app_test_list1.append(Appliance(channel,new_df_test1[[channel]],[0,0]))
#     app_test_list2.append(Appliance(channel,new_df_test2[[channel]],[0,0]))

for channel in columns:
    app_train_list.append(Appliance(channel,four_app_train[[channel]],[0,0]))
    app_test_list1.append(Appliance(channel,four_app_test[[channel]],[0,0]))


num_states_dict={}

ModelDict = {}
for i,app in enumerate(app_train_list):
    power_data_train = app.good_chunks.fillna(value = 0,inplace = False)
    X_train = power_data_train.values.reshape((-1, 1))
    power_data_test = app_test_list1[i].power_data.fillna(value = 0,inplace = False)
    X_test = power_data_test.values.reshape((-1, 1))
    hmm = HMM(X_train,X_test)
    hmm.fit_HMM()
    ModelDict[app.name] = hmm.model
    means = hmm.model.means_.round().astype(int).flatten().tolist()
    percMAD = hmm.HMM_MAD_perc(X_test, means)
    num_states_dict[app.name] = hmm.n_states
    print "MAD % for {} with {} states: {}".format(app.name, hmm.n_states, percMAD)


# def check_working(x):
#     return 1 if x>1 else 0
#
# four_app_test['apps_working'] = four_app_test['channel_12'].apply(lambda x: check_working(x)) + \
#                                  four_app_test['channel_5'].apply(lambda x: check_working(x)) + \
#                                  four_app_test['channel_6'].apply(lambda x: check_working(x)) + \
#                                  four_app_test['channel_7'].apply(lambda x: check_working(x))
#
# all_working = four_app_test['2013-08-01 11:40:00':'2013-08-01 17:00:00']

fhmm = FHMM.FHMM()
fhmm.train(app_train_list,num_states_dict = num_states_dict)
predictions = pd.DataFrame()
predictions = fhmm.disaggregate(four_app_test[['total']], predictions)


total_power_predicted = predictions.sum()
total_power_act = four_app_test[predictions.columns].sum()

print total_power_predicted
print total_power_act

print total_power_predicted.sort_index().values/total_power_act.sort_index().values




# for shift in range(len(all_working)/20):
#     subset = all_working[20*shift:20*shift+20]
#     x_test = subset[['total']].values.reshape((-1, 1))
#     prediction = fhmm.disaggregate_chunk(subset[['total']])
    # counter = 0
    # for model in ModelDict:
    #     logprob = ModelDict[model].score(x_test)
    #     if counter == 0:
    #         best_fit = ModelDict[model]
    #         best_score = logprob
    #     else:
    #         if logprob > best_score:
    #             best_fit = ModelDict[model]
    #             best_score = logprob
    #     counter +=1
    # chunk_MAD = HMM_MAD(best_fit,x_test)



# for model in ModelDict:


# for ind in fhmm.individual:
#     print fhmm.individual[ind].transmat_
#     print "*"*100

# print fhmm.model.transmat_


