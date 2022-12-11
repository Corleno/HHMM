import argparse
import os
### Libraries ###
import pickle
import time
import numpy as np
from matfuncs import expm
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import pandas as pd


def Initialization(verbose=False):
    global ages_test, testTypes_test, observations_test, treatment_indx_test, censor_ages_test, death_state_test, label_test, ind, nTests, inv, n_inv, MPmatrixs, nPatients_test
    global out_path, out_folder, max_steps_em, max_steps_optim, model, autograd_optim, p, args, n_test
    global currPars, curr_parameter_vector, threshold_test

    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass

    if args.test:
        out_folder = args.name + '_' + str(
            args.min_age) + '_' + args.dataset + '_' + args.age_inv + '_' + args.model + '_test'
    else:
        out_folder = args.name + '_' + str(args.min_age) + '_' + args.dataset + '_' + args.age_inv + '_' + args.model

    dataset = args.dataset
    model = args.model
    p = args.Z_prior

    if dataset == 'data_random_240k':
        data_location = '../../data/data_random_240k/'
    elif dataset == 'data_random':
        data_location = '../../data/data_random/'
    elif dataset == 'data_1000':
        data_location = '../../data/data_1000/'
    elif dataset == "data_random_240k_test":
        data_location = '../../data/data_random_240k_test/'
    else:
        print("dataset {} is not available".format(dataset))

    # load data
    testTypes = pickle.load(open(data_location + "mcmcPatientTestTypes", 'rb'), encoding="bytes")
    observations = pickle.load(open(data_location + "mcmcPatientObservations", 'rb'), encoding="bytes")
    ages = pickle.load(open(data_location + "mcmcPatientAges", 'rb'), encoding="bytes")
    treatment_indx = pickle.load(open(data_location + "mcmcPatientTreatmentIndx", 'rb'), encoding="bytes")
    censor_ages = pickle.load(open(data_location + "mcmcPatientCensorDates", 'rb'), encoding="bytes")
    death_states = pickle.load(open(data_location + "mcmcPatientDeathStates", 'rb'), encoding="bytes")
    birth_dates = pickle.load(open(data_location + "mcmcPatientBirthdates", "rb"))

    # Testing data
    N = len(testTypes)
    testTypes_test = list()
    observations_test = list()
    ages_test = list()
    treatment_indx_test = list()
    censor_ages_test = list()
    death_state_test = list()
    threshold_test = list()

    for i in range(N):
        threshold = cohort_range[0]
        n0 = np.sum(ages[i] < threshold)
        if ages[i][0] < threshold:
            temp_obs = observations[i][:n0]
            if birth_dates[i].year > (2012 - cohort_range[0]) or birth_dates[i].year < (2002 - cohort_range[1]):
                continue
            if np.sum(temp_obs, axis=0)[:2, -1].sum() > 0:
                continue
            if censor_ages[i] <= cohort_range[0]:
                continue
            threshold_test.append(threshold)
            testTypes_test.append([testTypes[i][:n0], testTypes[i][n0:]])
            observations_test.append([observations[i][:n0], observations[i][n0:]])
            ages_test.append([ages[i][:n0], ages[i][n0:]])
            # import pdb; pdb.set_trace()
            treatment_indx_test.append([np.array(treatment_indx[i])[np.array(treatment_indx[i]) < n0],
                                        np.array(treatment_indx[i])[np.array(treatment_indx[i]) >= n0] - n0])
            censor_ages_test.append(censor_ages[i])
            death_state_test.append(death_states[i])

    nPatients_test = len(ages_test)
    print('Number of patients for testing: ', nPatients_test)

    return 0


def Risk_stratification(version):
    patient_features = list()
    if version == "empirical":
        no_frailty = list()
        frailty = list()
    for indx in range(nPatients_test):
        if indx % 100 == 99:
            print("{}/{} has been completed".format(indx + 1, nPatients_test))
        # patient_ages = ages_test[indx][0]
        # patient_tests = testTypes_test[indx][0]
        patient_observations = observations_test[indx][0]
        # patient_treatment_indx = treatment_indx_test[indx][0]
        # patient_censor_age = threshold
        # patient_death_state = 0
        if version == "empirical":
            temp_C_H = np.mean(patient_observations, axis=0)[:2, :]
            temp_HPV = np.mean(patient_observations, axis=0)[2,:2]
            if np.sum(temp_C_H[:, 1:]) > 0 or np.sum(temp_HPV[1]) > 0:
                frailty.append(indx)
            else:
                no_frailty.append(indx)

    if version == "empirical":
        return np.array(no_frailty), np.array(frailty)

    ### Draw the empirical Kaplan Meier curve


def Draw_observations(ages, observations, censors, thresholds, KM_option = 0):
    N = len(ages)
    times = np.zeros(N)
    censoring = np.zeros(N)
    for patient_index, patient_ages, patient_observations, patient_censor, patient_threshold in zip(np.arange(len(ages)), ages, observations, censors, thresholds):
        times[patient_index] = patient_censor - patient_threshold
        for patient_age, patient_observation in zip(patient_ages, patient_observations):
            if Option_KM(patient_observation, KM_option):
                times[patient_index] = patient_age - patient_threshold
                censoring[patient_index] = 1
                break
    return times, censoring


### Option with respect ot the definition of Kaplan Meier estimators
def Option_KM(patient_observation, KM_option):
    if KM_option == 0:  # failure measured from observed 0 to observed (1,2,3).  Ignore HPV.
        if np.sum(patient_observation[:2, 1:]) > 0:
            return 1
        else:
            return 0

    if KM_option == 1:  # failure measured from observed (0,1) to observed (2,3).  Ignore HPV.
        if np.sum(patient_observation[:2, 2:]) > 0:
            return 1
        else:
            return 0


def main(index_range=0):
    global cohort_range
    #################################
    ####### Initialization ##########
    #################################
    cohort_ranges = [[30, 34], [35, 39], [40, 44], [45, 49], [50, 54], [55, 59], [60, 64], [65, 69]]
    cohort_range = cohort_ranges[index_range]
    version = 'empirical'
    Initialization()

    ###############################################
    ######## Empirical Risk Stratification ########
    ###############################################
    if version == 'empirical':
        no_frailty_indexes, frailty_indexes = Risk_stratification(version)
        L_indexes = no_frailty_indexes
        H_indexes = frailty_indexes

    ########################################
    ######## Plot empirical KM curve #######
    ########################################
    # low risk patients
    L_ages = [ages_test[index][1] for index in L_indexes]
    L_observations = [observations_test[index][1] for index in L_indexes]
    L_censors = [censor_ages_test[index] for index in L_indexes]
    L_time, L_censoring = Draw_observations(L_ages, L_observations, L_censors, threshold_test, KM_option=1)
    # low risk patients
    H_ages = [ages_test[index][1] for index in H_indexes]
    H_observations = [observations_test[index][1] for index in H_indexes]
    H_censors = [censor_ages_test[index] for index in H_indexes]
    H_time, H_censoring = Draw_observations(H_ages, H_observations, H_censors, threshold_test, KM_option=1)

    data = {'index': np.concatenate([L_indexes, H_indexes]), 'time': np.concatenate([L_time, H_time]),
            'censoring': np.concatenate([L_censoring, H_censoring]),
            'group': np.concatenate(
                [np.repeat("low risk", L_indexes.shape[0]), np.repeat("high risk", H_indexes.shape[0])])}
    # data = {'time': np.concatenate([L_time, H_time, L_time, H_time]), 'censoring': np.concatenate([L_censoring,
    #          H_censoring, L_censoring, H_censoring]), 'group': np.concatenate([np.repeat("low risk", L_indexes.shape[0]),
    #          np.repeat("high risk", H_indexes.shape[0]), np.repeat("entire", L_indexes.shape[0] + H_indexes.shape[0])])}
    df = pd.DataFrame(data, columns=['index', 'time', 'censoring', 'group'])
    # import pdb; pdb.set_trace()

    import kaplanmeier as km
    out = km.fit(df['time'], df['censoring'], df['group'])

    # save results
    if version == 'empirical':
        with open("model_res/{}/cohort_range_{}_{}_baseline_v{}.pickle".format(args.dataset, cohort_range[0], cohort_range[1], version),
                  "wb") as res:
            pickle.dump(df, res)

    # fig = plt.figure()
    # km.plot(out)
    # plt.tight_layout()
    # plt.savefig("threshold_{}_baseline.png".format(threshold))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of the experiment", default="clinical_application")
    parser.add_argument("--min_age", help="minimum age", type=int, default=16)
    parser.add_argument("--dataset",
                        help="data specification: data_1000, data_random and data_random_240k are available",
                        default="data_random_240k_test")
    parser.add_argument("--age_inv", help="age interval sepecification", default="inv4")
    parser.add_argument("--model", help="discrete or continuous model", default="continuous")
    parser.add_argument("--test", help="boolean indicator for testing", action='store_true')
    parser.add_argument("--Z_prior", help="prior probability of Model 1", type=np.float32, default=0.2)

    args = parser.parse_args()

    for index_range in range(8):
        main(index_range)
