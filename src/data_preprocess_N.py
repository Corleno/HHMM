#!/usr/bin/python3
# date: 07/26/2019

import pickle
import time
import os
import numpy as np

if __name__ == "__main__":

    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass

    location = '../data/data_1000/'

    # 1. mcmcPatientTestTypes: testTypes[p] patient_tests, patient_tests[j] patient_test ([3,], 3 categories)
    # 2. mcmcPatientObservations: observations[p] patient_observations, patient_observations[j] patient_observation ([3,4], count)
    # 3. mcmcPatientAges: ages[p] patient_ages, patient_ages[j] patient_age (scaler)

    # 4. mcmcPatientTreatmentIndx: treatment_indx[p] patient_treatment_indx (array)
    # 5. mcmcPatientCensorDates: censor_ages[p] censor_age (scalar)
    # 6. mcmcPatientDeathStates: death_states[p] death_state ([0,1])

    testTypes = pickle.load(open(location + "mcmcPatientTestTypes", 'rb'), encoding="bytes")
    observations = pickle.load(open(location + "mcmcPatientObservations", 'rb'), encoding="bytes")
    ages = pickle.load(open(location + "mcmcPatientAges", 'rb'), encoding="bytes")
    treatment_indx = pickle.load(open(location + "mcmcPatientTreatmentIndx", 'rb'), encoding="bytes")
    censor_ages = pickle.load(open(location + "mcmcPatientCensorDates", 'rb'), encoding="bytes")
    death_states = pickle.load(open(location + "mcmcPatientDeathStates", 'rb'), encoding="bytes")

    n_Patients = len(ages)
    print("Number of patients: {}".format(n_Patients))

    # print(testTypes[0])
    # print(observations[0])
    # print(ages[0])
    # print(treatment_indx[0])
    # print(censor_ages[0])
    # print(death_states[0])

    indx = 0
    features = []
    output = []
    feature_size = 1 + 1  # age + treatment
    output_size = 10  # observations

    for patient_test, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state in zip(
            testTypes, observations, ages, treatment_indx, censor_ages, death_states):
        n_visiting = len(patient_ages)
        patient_features = np.zeros([feature_size, n_visiting])
        patient_output = np.zeros([output_size, n_visiting])
        if not patient_treatment_indx:
            patient_treatment_indx_first = np.inf
        else:
            patient_treatment_indx_first = np.array(patient_treatment_indx[0])
        for n in range(n_visiting):
            patient_features[:, n] = np.concatenate([[patient_ages[n]], [float(n >= patient_treatment_indx_first)]])
            patient_output[:, n] = patient_observations[n].reshape(-1)[:10]
        features.append(patient_features)
        output.append(patient_output)
        # (float(patient_observations[n_visiting - 1][:2, -2:].sum() > 0))

    print(len(features))
    print(len(output))
    with open("../data/data_1000/vectorized_data_N.pickle", "wb") as res:
        pickle.dump([features, output], res)

    # print(features[0], objective[0])
    # Split training and testing by 8:2
    from sklearn.model_selection import train_test_split

    # X_train, X_test, y_train, y_test = train_test_split(features, objective, test_size = 0.2, random_state=22)
    X_train = features[:80000]
    Y_train = output[:80000]
    X_test = features[-20000:]
    Y_test = output[-20000:]
    with open("../data/data_1000/vectorized_data_trainandtest_N.pickle", "wb") as res:
        pickle.dump([X_train, X_test, Y_train, Y_test], res)
