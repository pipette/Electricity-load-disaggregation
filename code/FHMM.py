import pandas as pd
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import cPickle as pk
import itertools
from hmmlearn.hmm import GaussianHMM
from collections import OrderedDict
from copy import deepcopy
from Clustering import cluster

def compute_A_fhmm(list_A):
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result

def combine_parameters(ind_parameters_list):
    '''
    Function to compute factorized HMM parameters such as starting probabilities or transition matrices
    from individual model parameters
    :param ind_parameters_list: parameter list for individual models (per appliance), e.g. list of starting probabilites,
    transition matrices
    :return: matrix of factorized model parameters
    '''
    FactAParam = ind_parameters_list[0]
    for idx in xrange(1,len(ind_parameters_list)):
        FactA = np.kron(FactAParam,ind_parameters_list[idx])
    return FactAParam

def compute_pi_fhmm(list_pi):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs
    Returns
    -------
    result : Combined Pi for the FHMM
    """
    result = list_pi[0]
    for i in range(len(list_pi) - 1):
        result = np.kron(result, list_pi[i + 1])
    return result

def combine_means(means_list):
    """
    Compute factorized model states means
    :param means_list: list of individual models' state means
    :return: column vector of fatorized state means
    """
    states_combination = list(itertools.product(*means_list))
    num_combinations = len(states_combination)
    means_stacked = np.array([sum(x) for x in states_combination])
    means = np.reshape(means_stacked, (num_combinations, 1))
    cov = np.tile(5 * np.identity(1), (num_combinations, 1, 1))
    return [means, cov]

def sort_startprob(mapping, startprob):
    """ Sort the startprob according to power means; as returned by mapping
    """
    num_elements = len(startprob)
    new_startprob = np.zeros(num_elements)
    for i in xrange(len(startprob)):
        new_startprob[i] = startprob[mapping[i]]
    return new_startprob


def sort_covars(mapping, covars):
    new_covars = np.zeros_like(covars)
    for i in xrange(len(covars)):
        new_covars[i] = covars[mapping[i]]
    return new_covars


def sort_transition_matrix(mapping, A):
    """Sorts the transition matrix according to increasing order of
    power means; as returned by mapping
    Parameters
    ----------
    mapping :
    A : numpy.array of shape (k, k)
        transition matrix
    """
    num_elements = len(A)
    A_new = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            A_new[i, j] = A[mapping[i], mapping[j]]
    return A_new


def sort_learnt_parameters(startprob, means, covars, transmat):
    mapping = return_sorting_mapping(means)
    means_new = np.sort(means, axis=0)
    startprob_new = sort_startprob(mapping, startprob)
    covars_new = sort_covars(mapping, covars)
    transmat_new = sort_transition_matrix(mapping, transmat)
    assert np.shape(means_new) == np.shape(means)
    assert np.shape(startprob_new) == np.shape(startprob)
    assert np.shape(transmat_new) == np.shape(transmat)

    return [startprob_new, means_new, covars_new, transmat_new]


def return_sorting_mapping(means):
    means_copy = deepcopy(means)
    means_copy = np.sort(means_copy, axis=0)

    # Finding mapping
    mapping = {}
    for i, val in enumerate(means_copy):
        mapping[i] = np.where(val == means)[0][0]
    return mapping

def create_combined_hmm(model):
    list_pi = [model[appliance].startprob_ for appliance in model]
    list_A = [model[appliance].transmat_ for appliance in model]
    list_means = [model[appliance].means_.flatten().tolist()
                  for appliance in model]

    pi_combined = compute_pi_fhmm(list_pi)
    A_combined = compute_A_fhmm(list_A)
    [mean_combined, cov_combined] = combine_means(list_means)

    combined_model = GaussianHMM(
        n_components=len(pi_combined), covariance_type='full',
        startprob_prior=pi_combined, transmat_prior=A_combined)
    combined_model.covars_ = cov_combined
    combined_model.means_ = mean_combined
    combined_model.startprob_ = pi_combined
    combined_model.transmat_ = A_combined
    return combined_model


def decode_hmm(length_sequence, centroids, appliance_list, states):
    """
    Decodes the HMM state sequence
    """
    hmm_states = {}
    hmm_power = {}
    total_num_combinations = 1

    for appliance in appliance_list:
        total_num_combinations *= len(centroids[appliance])

    for appliance in appliance_list:
        hmm_states[appliance] = np.zeros(length_sequence, dtype=np.int)
        hmm_power[appliance] = np.zeros(length_sequence)

    for i in range(length_sequence):

        factor = total_num_combinations
        for appliance in appliance_list:
            # assuming integer division (will cause errors in Python 3x)
            factor = factor // len(centroids[appliance])

            temp = int(states[i]) / factor
            hmm_states[appliance][i] = temp % len(centroids[appliance])
            hmm_power[appliance][i] = centroids[
                appliance][hmm_states[appliance][i]]
    return [hmm_states, hmm_power]

class FHMM():
    """
    Attributes
    ----------
    model : dict
    predictions : pd.DataFrame()
    meters : list
    MIN_CHUNK_LENGTH : int
    """

    def __init__(self):
        self.model = {}
        self.predictions = pd.DataFrame()
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = 'FHMM'


    def train(self, appliances, num_states_dict={}, **load_kwargs):
        """Train using 1d FHMM.
        Places the learnt model in `model` attribute
        The current version performs training ONLY on the first chunk.
        Online HMMs are welcome if someone can contribute :)
        Assumes all pre-processing has been done.
        """
        learnt_model = OrderedDict()
        num_meters = len(appliances)
        if num_meters > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3

        for i, app in enumerate(appliances):
            power_data = app.power_data.fillna(value = 0,inplace = False)
            X = power_data.values.reshape((-1, 1))
            assert X.ndim == 2
            self.X = X

            if num_states_dict.get(app.name) is not None:
                # User has specified the number of states for this appliance
                num_total_states = num_states_dict.get(app.name)

            else:
                # Find the optimum number of states
                print "Identifying number of hidden states for appliance {}".format(app.name)
                states = cluster(X, max_num_clusters)
                num_total_states = len(states)
                print "Number of hidden states for appliance {}: {}".format(app.name, num_total_states)

            print("Training model for appliance '{} with {} hidden states'".format(app.name, num_total_states))
            learnt_model[app.name] = GaussianHMM(num_total_states, "full")

            # Fit
            learnt_model[app.name].fit(X)


        # Combining to make a AFHMM
        self.meters = []
        new_learnt_models = OrderedDict()
        for app in learnt_model:
            startprob, means, covars, transmat = sort_learnt_parameters(
                learnt_model[app].startprob_, learnt_model[app].means_,
                learnt_model[app].covars_, learnt_model[app].transmat_)
            new_learnt_models[app] = GaussianHMM(
                startprob.size, "full")
            new_learnt_models[app].means_ = means
            new_learnt_models[app].covars_ = covars
            new_learnt_models[app].startprob_ = startprob
            new_learnt_models[app].transmat_ = transmat
            # UGLY! But works.
            self.meters.append(app)

        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined

    def disaggregate_chunk(self, test_mains):
        """Disaggregate the test data according to the model learnt previously
        Performs 1D FHMM disaggregation.
        For now assuming there is no missing data at this stage.
        """

        # Array of learnt states
        learnt_states_array = []
        test_mains = test_mains.dropna()
        length = len(test_mains.index)
        temp = test_mains.values.reshape(length, 1)
        learnt_states_array.append(self.model.predict(temp))

        # Model
        means = OrderedDict()
        for elec_meter, model in self.individual.iteritems():
            means[elec_meter] = (
                model.means_.round().astype(int).flatten().tolist())
            means[elec_meter].sort()

        decoded_power_array = []
        decoded_states_array = []

        for learnt_states in learnt_states_array:
            [decoded_states, decoded_power] = decode_hmm(
                len(learnt_states), means, means.keys(), learnt_states)
            decoded_states_array.append(decoded_states)
            decoded_power_array.append(decoded_power)

        prediction = pd.DataFrame(
            decoded_power_array[0], index=test_mains.index)

        return prediction


    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.
        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        sample_period : number, optional
            The desired sample period in seconds.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        import warnings
        warnings.filterwarnings("ignore", category=Warning)

        for chunk in mains.power_series(**load_kwargs):
            # Check that chunk is sensible size before resampling
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue

            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            # Start disaggregation
            predictions = self.disaggregate_chunk(chunk)
            for meter in predictions.columns:

                meter_instance = meter.instance()
                cols = pd.MultiIndex.from_tuples([chunk.name])
                predicted_power = predictions[[meter]]
                if len(predicted_power) == 0:
                    continue
                data_is_available = True
                output_df = pd.DataFrame(predicted_power)
                output_df.columns = pd.MultiIndex.from_tuples([chunk.name])
                key = '{}/elec/meter{}'.format(building_path, meter_instance)
                output_datastore.append(key, output_df)

            # Copy mains data to disag output
            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))

        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=self.meters
            )

