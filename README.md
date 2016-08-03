# Smart meter load disaggregation

##Motivation

The smart meter usage has been increasing in the past years as people get more motivated to improve their energy management, reduce consumption and utility costs.
But even though smart meters help us understand our aggregate power consumption, we are still poor at estimating how much energy individual appliances are
using. This can lead to suboptimal choices when deciding what appliance to turn off or replace.

This is when smart meter disaggregation (also known as Non-intrusive load monitoring or NILM) can be useful. To put simply smart meter disaggreggation is a process of 
 extracting individual appliance power signatures from the total aggregate signal. It can be argued that directly measuring individual appliance consumption by 
means of installing smart plugs would be easier and lead to more precise measurements. While this is conceptually true, installing smart plugs is an expensive and cumbersome 
process and considering how many appliances an average household uses this solution will simply not scale.

All that being said, the basic aim of the smart meter disaggregation algorithm can be stated as follows:

- On one hand given the total smart meter signal, can we infer what appliances generated that signal
- And on the other hand can we infer the quantity of energy used by each appliance.

These are the two questions I tried to answer in project.


##Data
I  used the May 2016 release for UK domestic appliance-level ([UK-DALE](http://www.doc.ic.ac.uk/~dk3810/data/)) electricity dataset. 
The dataset contains power demand for 5 houses, where for each house they measure whole house mains power demand as well as appliance level demand
 at 6 seconds frequency.
 For three of the six houses they also provide voltage and current measurements but for the time being I just focused on power.
 The meter readings for each appliance (labeled as channel in the dataset) are stored in separate .dat files which are easy to load into python with pandas.
After loading the data into pandas I resampled it into 1 min frequencies and used 2 months of data for training and two sets of 1 month
of observations for testing. 


##Methodology

### The model
From the very beginning I decided that I'm only going to use Hidden Markov Model to extract the appliance signal. HMM proved to perform well in such 
tasks as on-line handwriting and speech recognition. Electricity signal can be in certain sense compared to these two taks since we are dealing 
with sequential data and one can expect a correlation between two points that are close in the sequence.
The only problem with this approach is that in the case of canonical HMM we have one process with multiple states generating the observed signal. 
In case of electricity signal we have multiple appliances each with multiple states generating observed output. A good solution to this problem is a 
extension to a standard HMM approach known as factorial Hidden Markov Model. 
In a factorial markov model we have **M** independent Markov chains of latent variables each of which can be in **K** states. 
The distribution of the observed variable at
a given time step is conditional on the states of all of the corresponding latent variables at that same time step.
The current implementation uses what is known as exact factorial HMM. The major problem with this approach is that it has computational
 complexity O(NK2M) that is exponential in the number M of latent chains and so will be intractable for anything other than small values of M.
For this reason I focused on four appliances that generated the highest power demand for house #1 over one year. 
I treated the number of states for each latent chain as a hyperparameter and found 3 to 4 states for each chain to produce the best results.

The code is written in Python and makes use of the following libraries:

* [Pandas]: Python data analysis toolkit
* [HMMLearn]: algorithms and models to learn HMM in python
* [SKLearn]: Python's Machine Learning library

[Pandas]: http://pandas.pydata.org/
[HMMLearn]: https://hmmlearn.readthedocs.io/en/latest/
[SKLearn]: http://scikit-learn.org/stable/

### Performance Accuracy Metrics
Two main points that we have to keep in mind when deciding which accuracy metric to use are:

- an algorithm needs to accuratly predict when a particular device is ON vs OFF
- an algorithm needs to accuratly predict the amount of energy used at a particular point in time.

Taking both of these points into consideration Mean Absolute Deviation (MAD) or a simple R2 score both seem to be reasonable metrics to use, as they
both capture when device is ON vs OFF as well as how far the estimate deviates from the baseline. I finally settled for R2 as it is easier to
interpret and provides another interesting insight, namely if the coefficient is negative it tells us that observed and predicted data are a poor match or not in agreement.
MAD will not provide us this information.

###Model Evaluation
For this project I calculated an individual R2 for each of the four appliances, however I can imagine going forward that as the number of appliances 
increases a single metric would be more usefull.
Since we are dealing with time series, we can't just randomly split the data into training and testing set. For this reason I used 2 months of data 
for training and 2 sets of 1 month of observations for testing. One testing set was used to validate individual HMM models and the second to validate the
factorial HMM.

The table and graphs below show the performance of the best model on the test set.

|  Water heater  |   Dishwasher   |     Fridge      | Washing machine |
|:--------------:|:--------------:|:---------------:|:---------------:|
|        0.84    |       0.92     |       0.81      |       0.91      |


<img src="/img/fridge_profile.png" width="450" height="350" />


<img src="/img/kettle.png" width="450" height="350"/>


##Take-aways
The examples below show how disaggregated data can be used by both utilities and their customers to save money and also reduce carbon footprint.
On the customer side splitting the total usage up into components provides better insights which appliance uses most of their energy,
 users can compare to other households or over time. So they can make better decisions how to optimise their use, reduce cost or CO2.
 
<img src="/img/monthly.png" width="450" height="350" />

Hourly information can also be used by utilities to identify customers like the one below. We can see that part of their high demand can 
easily be shifted from peak to off-peak hours. So energy providers  can give incentives or time-of-use pricing to customers to optimize grid load an 
save on imbalance prices. 

<img src="/img/3_days_ind.png" width="450" height="350" />


##Conclusion


The scope of this project had to be limited due to time constraints (2 weeks) but can be continued in the following directions:

- implementing variational methods instead of exact factorial HMM. This will allow to add more appliances for a single house and also generalize better accross buildings;
- there is a great dataset recording appliance electricity demand for different appliance types and modes (e.g. washing machine operating at 30C vs 60C).
It would be interesting to train on all of those appliances and see if the signal be extracted accurately from the aggregated smart meter signal;
- incorporating housold specific, building specific and weather data into the model;
- extending the model to short-term (15-30 min) forecasting.