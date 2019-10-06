#!/usr/bin/env python3
"""
Prediction for HHMM
Most recent version as of 07/28/2019
"""

import argparse
import os
# import logging
import sys
### Libraries ###
import pickle
import time
import numpy as np
from matfuncs import expm
from scipy import stats
from scipy import special
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def Initialization():
    global ages_test, testTypes_test, observations_test, treatment_indx_test, censor_ages_test, death_state_test, ind, nTests, inv, n_inv, MPmatrixs, nPatients_test
    global out_path, out_folder, max_steps_em, max_steps_optim, model, autograd_optim, p, args, n_test
    global currPars, curr_parameter_vector

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of the experiment", default="prediction_hierarchical")
    parser.add_argument("--min_age", help="minimum age", type=int, default=16)
    parser.add_argument("--dataset", help="data specification: data_1000 are available", default="data_1000")
    parser.add_argument("--age_inv", help="age interval sepecification", default="inv4")
    parser.add_argument("--model", help="discrete or continuous model", default="continuous")
    parser.add_argument("--test", help="boolean indicator for testing", action='store_true')
    parser.add_argument("--Z_prior", help="prior probability of Model 1", type=np.float32, default=0.5)

    args = parser.parse_args()

    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass

    if args.test:
        out_folder = args.name + '_' + str(
            args.min_age) + '_' + args.dataset + '_' + args.age_inv + '_' + args.model + '_test'
    else:
        out_folder = args.name + '_' + str(args.min_age) + '_' + args.dataset + '_' + args.age_inv + '_' + args.model

    min_age = args.min_age
    do_truncate_ages = True if min_age > 16 else False
    dataset = args.dataset
    inv_indx = args.age_inv
    model = args.model
    p = args.Z_prior

    ##########################
    ##### Initialization #####
    ##########################
    nStates_0 = 2
    nStates_1 = 4
    nTests = 3

    ind = ["Alpha", "Eta", "W",
           "C"]  # ind specifies which parameters need to be optimized:x ["Alpha", "W", "Gamma", "Zeta", "Eta", "A", "C"]
    inv_list = {"inv14": [20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 53, 57, 60, 200],
                "inv13": [20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 55, 60, 200],
                "inv12": [20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 60, 200],
                "inv11": [20, 23, 27, 30, 33, 37, 40, 45, 50, 60, 200],
                "inv10": [19, 22, 25, 30, 35, 40, 45, 50, 55, 200],  # expert advice
                "inv9": [23, 25, 30, 35, 40, 45, 50, 60, 200],  # expert advice
                "inv8": [20, 25, 30, 35, 40, 50, 60, 200],  # best AIC and Likelihood with 3000 procs.
                "inv7": [20, 25, 30, 40, 50, 60, 200],  # best AIC with 300 procs
                "inv6": [23, 30, 40, 50, 60, 200],
                "inv5": [23, 35, 45, 60, 200],
                "inv4": [23, 30, 60, 200],
                "inv3": [29, 69, 200],  # close second AIC with 300 procs
                "inv2": [23, 200],
                "inv1": [200]}
    inv = inv_list[inv_indx]
    n_inv = len(inv)

    if dataset == 'data_1000':
        data_location = '../../data/data_1000/'
    else:
        print("dataset {} is not available".format(dataset))

    # load data
    testTypes = pickle.load(open(data_location + "mcmcPatientTestTypes", 'rb'), encoding="bytes")
    observations = pickle.load(open(data_location + "mcmcPatientObservations", 'rb'), encoding="bytes")
    ages = pickle.load(open(data_location + "mcmcPatientAges", 'rb'), encoding="bytes")
    treatment_indx = pickle.load(open(data_location + "mcmcPatientTreatmentIndx", 'rb'), encoding="bytes")
    censor_ages = pickle.load(open(data_location + "mcmcPatientCensorDates", 'rb'), encoding="bytes")
    death_states = pickle.load(open(data_location + "mcmcPatientDeathStates", 'rb'), encoding="bytes")

    # Testing data
    # n_test: number of testing data
    n_test = 20000
    testTypes_test = testTypes[-n_test:]
    observations_test = observations[-n_test:]
    ages_test = ages[-n_test:]
    treatment_indx_test = treatment_indx[-n_test:]
    censor_ages_test = censor_ages[-n_test:]
    death_state_test = death_states[-n_test:]

    # define Markov Process topology with MPmatrix. The diagonal should be zeros.
    # A one in element (i,j) indicates a possible transition between states i and j.
    MPmatrix_0 = np.zeros([nStates_0 + 1, nStates_0 + 1])
    MPmatrix_0[0, 1] = MPmatrix_0[1, 0] = MPmatrix_0[0, 2] = MPmatrix_0[1, 2] = 1
    MPmatrix_1 = np.zeros([nStates_1 + 1, nStates_1 + 1])
    MPmatrix_1[0, 1] = MPmatrix_1[1, 0] = MPmatrix_1[1, 2] = MPmatrix_1[2, 1] = MPmatrix_1[2, 3] = MPmatrix_1[:-1,
                                                                                                   -1] = 1
    MPmatrixs = [MPmatrix_0, MPmatrix_1]

    nPatients_test = len(ages_test)
    print('Number of patients for testing: ', nPatients_test)

    ### Set informative initial parameters
    temp = 4
    currAlpha_0 = [np.zeros([nStates_0, 4]), np.zeros([nStates_0, 4]), np.zeros([nStates_0, 2])]
    currAlpha_0[0][0, 0] = currAlpha_0[0][1, 1] = temp
    currAlpha_0[1][0, 0] = currAlpha_0[1][1, 1] = temp
    currAlpha_1 = [np.zeros([nStates_1, 4]), np.zeros([nStates_1, 4]), np.zeros([nStates_1, 2])]
    currAlpha_1[0][0, 0] = currAlpha_1[0][1, 1] = currAlpha_1[0][2, 2] = currAlpha_1[0][3, 3] = temp
    currAlpha_1[1][0, 0] = currAlpha_1[1][1, 1] = currAlpha_1[1][2, 2] = currAlpha_1[1][3, 3] = temp
    currAlpha_1[2][3, 0] = -2
    currAlpha_1[2][3, 1] = 2
    currAlpha = [currAlpha_0, currAlpha_1]
    currEta_0 = np.zeros([nStates_0, 3])
    currEta_1 = np.zeros([nStates_1, 3])
    currEta = [currEta_0, currEta_1]
    if model == "continuous":
        currW_0 = np.zeros([4, n_inv])
        currW_0[1, :] = -temp
        currW_0[3, :] = -temp
        currW_1 = np.zeros([9, n_inv])
        currW_1[1, :] = -temp
        currW_1[4, :] = -temp
        currW_1[7, :] = -temp
    elif model == "discrete":
        currW_0 = -4 * np.ones([4, n_inv])
        currW_1 = -4 * np.ones([9, n_inv])
    currW = [currW_0, currW_1]
    currC_0 = np.zeros([nStates_0, n_inv])
    currC_1 = np.zeros([nStates_1, n_inv])
    currC = [currC_0, currC_1]

    return 0


def Load_EM_res(verbose=False):
    global currPars
    with open("../../data/data_2400/EM_16_updated_data_inv4_continuous_240000/res", "rb") as em_res:
        res = pickle.load(em_res, encoding="bytes")
    currAlpha = res[2]
    currEta = res[3]
    currW = res[4]
    currC = res[5]
    currPars = [currAlpha, currEta, currW, currC]

    if verbose:
        print(
            "EM results have been loaded with Pars: Alpha: {}, Eta: {}, W: {}, C:{}.".format(currAlpha, currEta, currW,
                                                                                             currC))
    return 0


def Compute_pos_Z_test(p, verbose=False):  ### given no state Z | -
    global Z_pos
    # It is a Bernouli(p)

    ts = time.time()
    Z_pos = []
    for indx in range(nPatients_test):
        if indx % 100 == 99:
            print("{}/{} has been completed".format(indx + 1, nPatients_test))
        loglik_0 = Loglikelihood_obs0_test(indx, 0, currPars)
        loglik_1 = Loglikelihood_obs0_test(indx, 1, currPars)
        tilde_p = np.exp(np.log(p) + loglik_1 - np.log((1 - p) * np.exp(loglik_0) + p * np.exp(loglik_1)))
        Z_pos.append(tilde_p)

    print('Compute the posterior of Z costs {}s'.format(time.time() - ts))
    if verbose:
        for indx in range(nPatients_test):
            print("Patient {}: model index probabilites: {}".format(indx, Z_pos[indx]))
    return 0


def Loglikelihood_obs0_test(indx, Z, Pars, verbose=False):
    Alpha = Pars[0][Z]
    Eta = Pars[1][Z]
    W = Pars[2][Z]
    C = Pars[3][Z]
    MPmatrix = MPmatrixs[Z]

    patient_ages = ages_test[indx]
    patient_tests = testTypes_test[indx]
    patient_observations = observations_test[indx]
    patient_treatment_indx = treatment_indx_test[indx]
    patient_censor_age = censor_ages_test[indx]
    patient_death_state = death_state_test[indx]

    if len(patient_treatment_indx) == 1:
        j = patient_treatment_indx[0]
        loglik0 = Loglikelihood_group_obs0(patient_tests[:(j + 1)], patient_observations[:(j + 1)],
                                           patient_ages[:(j + 1)], 0, patient_censor_age, patient_death_state, Z, Alpha,
                                           Eta, W, C, ind, inv, verbose)
        if j < (len(patient_ages) - 1):
            loglik1 = Loglikelihood_group_obs0(patient_tests[j:-1], patient_observations[j:-1], patient_ages[j:-1], 1,
                                               patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv,
                                               verbose)
        else:
            loglik1 = 0
        loglik = loglik0 + loglik1
    elif len(patient_treatment_indx) > 1:
        loglik = 0
        j = patient_treatment_indx[0]
        loglik0 = Loglikelihood_group_obs0(patient_tests[:(j + 1)], patient_observations[:(j + 1)],
                                           patient_ages[:(j + 1)], 0, patient_censor_age, patient_death_state, Z, Alpha,
                                           Eta, W, C, ind, inv, verbose)
        loglik += loglik0
        for i in range(len(patient_treatment_indx) - 1):
            j = patient_treatment_indx[i]
            k = patient_treatment_indx[i + 1] + 1
            logliki = Loglikelihood_group_obs0(patient_tests[j:k], patient_observations[j:k], patient_ages[j:k], 1,
                                               patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv,
                                               verbose)
            loglik += logliki
        j = patient_treatment_indx[-1]
        if j < (len(patient_ages) - 1):
            loglik1 = Loglikelihood_group_obs0(patient_tests[j:-1], patient_observations[j:-1], patient_ages[j:-1], 1,
                                               patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv,
                                               verbose)
        else:
            loglik1 = 0
        loglik += loglik1
    else:
        loglik = Loglikelihood_group_obs0(patient_tests[:-1], patient_observations[:-1], patient_ages[:-1], 0,
                                          patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv,
                                          verbose)
    return loglik


def Loglikelihood_group_obs0(patient_tests, patient_observations, patient_ages, patient_treatment_status, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose=False, do_last=False):
    if Z == 0:
        nStates = 2
    else:
        nStates = 4
    MPmatrix = MPmatrixs[Z]

    nvisits = len(patient_ages)
    ### Initialization ###
    ### Q[s] ~ Pr[S0=s, O0]
    Q = np.zeros(nStates)
    patient_age = patient_ages[0]
    patient_test = patient_tests[0]
    patient_observation = patient_observations[0]
    if patient_treatment_status == 0:
        for s in range(nStates):
            Q[s] = np.log(ddirichlet_categorical(s, np.exp(C[:, Age2Comp(patient_age, inv)])))
            # Q[s] = np.log(C[s, Age2Comp(patient_age, inv)])
            Q[s] += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[s, :])))
            # Q[s] += np.sum(stats.poisson.logpmf(patient_test, Eta[s,:]))
            for k in range(nTests):
                if k == 2:
                    # Q[s] += multinomial_logpmf(patient_observations[0][k, :2], Alpha[k][s, :])
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observations[0][k, :2], np.exp(Alpha[k][s, :])))
                else:
                    # Q[s] += multinomial_logpmf(patient_observations[0][k, :], Alpha[k][s, :])
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observations[0][k, :], np.exp(Alpha[k][s, :])))
            # P(S0, O0)
        Q = np.exp(Q)
    else:
        Q[0] = 1
    log_Q = np.log(Q)

    ####################
    ### Forward Pass ###
    ####################
    # P_forward_matrices P(Sj-1, Sj, O0-j)
    P_forward_matrices = [np.zeros([nStates, nStates]) for patient_age in patient_ages]
    for j in range(1, nvisits):
        p_transition = ProbTransition(MPmatrix, W, patient_ages[j - 1], patient_ages[j], inv)
        log_prob_obs = np.zeros(nStates)
        for s in range(nStates):
            log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], np.exp(Eta[s, :])))
            # log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], Eta[s,:]))
            for k in range(nTests):
                if k == 2:
                    # log_prob_obs[s] += multinomial_logpmf(patient_observations[j][k, :2], Alpha[k][s, :])
                    log_prob_obs[s] += np.log(
                        ddirichlet_mutilnominal(patient_observations[j][k, :2], np.exp(Alpha[k][s, :])))
                else:
                    # log_prob_obs[s] += multinomial_logpmf(patient_observations[j][k, :], Alpha[k][s, :])
                    log_prob_obs[s] += np.log(
                        ddirichlet_mutilnominal(patient_observations[j][k, :], np.exp(Alpha[k][s, :])))

        log_P_forward_matrix = np.repeat(log_Q, nStates).reshape([nStates, nStates]) + np.transpose(
            np.repeat(log_prob_obs, nStates).reshape([nStates, nStates])) + np.log(p_transition[:nStates, :nStates])
        P_forward_matrix = np.exp(log_P_forward_matrix)
        #
        P_forward_matrices[j] = P_forward_matrix
        #
        Q = np.sum(P_forward_matrix, 0) / np.sum(P_forward_matrix)
        log_Q = np.log(Q)

    ## P(S_T, O)
    if nvisits > 1:
        PP = np.sum(P_forward_matrices[nvisits - 1], 0)
    else:
        PP = Q
    log_PP = np.log(PP)

    # print ("P_forward_matrices", P_forward_matrices)
    # print ("log_PP", log_PP)

    ## P(S_T, S_last, O)
    if do_last:
        # Add the censor statue
        if patient_censor_age < patient_ages[-1]:
            # this can happen due to some rounding errors when death is very close to last screening.
            # Just move the censor date a few month after last visit.
            patient_censor_age = patient_ages[-1] + 0.25
        p_transition = ProbTransition(MPmatrix, W, patient_ages[-1], patient_censor_age, inv)
        if patient_death_state > 0:  # this means censor age is age of 'death', not end of observations.
            log_PP += np.log(p_transition[:nStates, -1])
        else:  # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
            log_PP += np.log(1. - p_transition[:nStates, -1])

    # print ("log_PP", log_PP)

    return np.log(np.sum(np.exp(log_PP)))


def Compute_pos_last2(Pars):
    global last2s

    ts = time.time()
    last2s = []
    for indx in range(nPatients_test):
        last2 = last2_z(indx, Pars, verbose=False)
        # print(last2_z0.sum(), last2_z1.sum())
        last2s.append(last2)
    # print(last2s)
    print('Compute the predictive distribution of S*_I costs {}s'.format(time.time() - ts))
    return 0


def last2_z(indx, Pars, verbose=False):
    Alpha = Pars[0]
    Eta = Pars[1]
    W = Pars[2]
    C = Pars[3]
    Z = 1

    patient_ages = ages_test[indx]
    patient_tests = testTypes_test[indx]
    patient_observations = observations_test[indx]
    patient_treatment_indx = treatment_indx_test[indx]
    patient_censor_age = censor_ages_test[indx]
    patient_death_state = death_state_test[indx]

    if len(patient_treatment_indx) >= 1:
        j = patient_treatment_indx[-1]
        res = prob_last2_z_group(patient_tests[:-1], patient_observations[:-1], patient_ages[:-1], 1,
                                 patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose)
    else:
        res = prob_last2_z_group(patient_tests[:-1], patient_observations[:-1], patient_ages[:-1], 0,
                                 patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose)

    if verbose:
        print(indx, patient_treatment_indx, patient_tests[:-1], patient_observations[:-1], patient_ages[:-1],
              patient_censor_age, patient_death_state)

    return res


def prob_last2_z_group(patient_tests, patient_observations, patient_ages, patient_treatment_status, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose=False):
    if Z == 0:
        nStates = 2
    else:
        nStates = 4
    MPmatrix = MPmatrixs[Z]

    nvisits = len(patient_ages)
    ### Initialization ###
    ### Q[s] ~ Pr[S0=s, O0]
    Q = np.zeros(nStates)
    patient_age = patient_ages[0]
    patient_test = patient_tests[0]
    patient_observation = patient_observations[0]
    if patient_treatment_status == 0:
        for s in range(nStates):
            Q[s] = np.log(ddirichlet_categorical(s, np.exp(C[:, Age2Comp(patient_age, inv)])))
            # Q[s] = np.log(C[s, Age2Comp(patient_age, inv)])
            Q[s] += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[s, :])))
            # Q[s] += np.sum(stats.poisson.logpmf(patient_test, Eta[s,:]))
            for k in range(nTests):
                if k == 2:
                    # Q[s] += multinomial_logpmf(patient_observations[0][k, :2], Alpha[k][s, :])
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observations[0][k, :2], np.exp(Alpha[k][s, :])))
                else:
                    # Q[s] += multinomial_logpmf(patient_observations[0][k, :], Alpha[k][s, :])
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observations[0][k, :], np.exp(Alpha[k][s, :])))
            # P(S0, O0)
        Q = np.exp(Q)
    else:
        Q[0] = 1
    log_Q = np.log(Q)

    ####################
    ### Forward Pass ###
    ####################
    # P_forward_matrices P(Sj-1, Sj, O0-j)
    P_forward_matrices = [np.zeros([nStates, nStates]) for patient_age in patient_ages]
    for j in range(1, nvisits):
        p_transition = ProbTransition(MPmatrix, W, patient_ages[j - 1], patient_ages[j], inv)
        log_prob_obs = np.zeros(nStates)
        for s in range(nStates):
            log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], np.exp(Eta[s, :])))
            # log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], Eta[s,:]))
            for k in range(nTests):
                if k == 2:
                    # log_prob_obs[s] += multinomial_logpmf(patient_observations[j][k, :2], Alpha[k][s, :])
                    log_prob_obs[s] += np.log(
                        ddirichlet_mutilnominal(patient_observations[j][k, :2], np.exp(Alpha[k][s, :])))
                else:
                    # log_prob_obs[s] += multinomial_logpmf(patient_observations[j][k, :], Alpha[k][s, :])
                    log_prob_obs[s] += np.log(
                        ddirichlet_mutilnominal(patient_observations[j][k, :], np.exp(Alpha[k][s, :])))

        log_P_forward_matrix = np.repeat(log_Q, nStates).reshape([nStates, nStates]) + np.transpose(
            np.repeat(log_prob_obs, nStates).reshape([nStates, nStates])) + np.log(p_transition[:nStates, :nStates])
        P_forward_matrix = np.exp(log_P_forward_matrix)
        P_forward_matrices[j] = P_forward_matrix
        Q = np.sum(P_forward_matrix, 0) / np.sum(P_forward_matrix)

    if nvisits == 1:
        Q /= Q.sum()

    return Q


def Compute_pos_last(Pars, verbose=False):
    global lasts

    ts = time.time()
    lasts = []
    Z = 1
    W = Pars[2]
    MPmatrix = MPmatrixs[Z]

    for indx in range(nPatients_test):
        last2_z1 = last2s[indx]
        patient_ages = ages_test[indx]

        # compute S_I+1|z=1, O*, hat_psi
        p_transition = ProbTransition(MPmatrix, W, patient_ages[-2], patient_ages[-1], inv)
        P_transition = p_transition[:-1, :-1]
        P_transition /= P_transition.sum(axis=1)[:, None]
        last_z1 = last2_z1.dot(P_transition)  # dim 4

        if verbose:
            print("{}th patient, last2_z1: {}, last_z1: {}".format(indx, last2_z1, last_z1))
        last = last_z1
        lasts.append(last_z1)

    if verbose:
        for indx in range(nPatients_test):
            print("{}th patient, state probs: {}, res: {}".format(indx, lasts[indx], observations_test[indx][-1]))
    print("Computing the last state probability costs {}s".format(time.time()-ts))


def ProbTransition_interval(MPmatrix, dt, W):
    '''
        'MPmatrix' should be a square N-by-N matrix of ones and zeros that defines the intensity matrix of the markov process.
        A 1 at element ij indicates a possible transition between states i and j.
        A 0 at element ij means no possible transition between states i and j.

        -- Because this is a continuous time Markov Process the diagonals are forced to be zero.
        -- 'lambdas' is an array of transition intensities for the given patient at a given time interval.
        -- dt is a scalar. It is the difference in time between two observations.

        '''
    matrix = np.array(MPmatrix, copy=True)

    if model == 'continuous':
        matrix_filled = np.zeros_like(matrix, dtype=np.float32)
        matrix_filled[np.where(matrix > 0)] = np.exp(W)
        for i in range(matrix.shape[0]):
            matrix_filled[i, i] = - np.sum(matrix_filled[i, :])
        out = expm(dt * matrix_filled)  # so far so good...
    elif model == 'discrete':
        n_dim = MPmatrix.shape[0]
        matrix_filled = np.zeros_like(matrix, dtype=np.float32)
        matrix_filled[np.where(matrix == 1)] = W
        np.fill_diagonal(matrix, 1)
        matrix = np.matmul(np.diag(1 + np.arange(n_dim)), matrix)
        for indx_row in range(n_dim):
            matrix_filled[np.where(matrix == 1 + indx_row)] = Softmax(matrix_filled[np.where(matrix == 1 + indx_row)])
        out = np.linalg.matrix_power(matrix_filled, int(round(dt * 12)) if int(
            round(dt * 12)) > 0 else 1)  # Assume the screening interval is at least one month

    # Normalize the probablity matrix
    out = np.where(out < 0, 0., out)
    out = np.where(out > 1, 1., out)
    norm = np.repeat(np.sum(out, 1), out.shape[0]).reshape(out.shape)
    out = out / norm
    return out


def ProbTransition(MPmatrix, W, start, end, inv):
    '''
        'matrix' should be a square N-by-N matrix of ones and zeros that defines the intensity matrix of the markov process.
        A 1 at element ij indicates a possible transition between states i and j.
        A 0 at element ij means no possible transition between states i and j.

        Because this is a continuous time Markov Process the diagonals are forced to be zero.

        hpv_status is 0,1 or -1. If -1, then status is unknown.
        treatment_status is 0 or 1.

    '''
    temp = start
    matrix = np.eye(MPmatrix.shape[0])

    while (temp < end):
        temp_component = Age2Comp(temp, inv)
        end_component = Age2Comp(end, inv)
        if temp_component < end_component:
            dt = (inv[temp_component] - temp)
            temp_W = W[:, temp_component]
            matrix = np.dot(matrix, ProbTransition_interval(MPmatrix, dt, temp_W))
            temp = inv[temp_component]
        else:
            dt = end - temp
            temp_W = W[:, temp_component]
            matrix = np.dot(matrix, ProbTransition_interval(MPmatrix, dt, temp_W))
            temp = inv[temp_component]

    out = matrix
    # Normalize the probability matrix
    out = np.where(out < 0, 0., out)
    out = np.where(out > 1, 1., out)
    norm = np.repeat(np.sum(out, 1), out.shape[0]).reshape(out.shape)
    out = out / norm
    return out


def Softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def ddirichlet_mutilnominal(x, alpha):
    n = np.sum(x)
    alpha0 = sum(alpha)
    if n == 0:
        return 1
    else:
        return n * special.beta(alpha0, n) / np.prod(
            np.array([special.beta(alphak, xk) * xk for alphak, xk in zip(alpha, x) if xk > 0]))


def ddirichlet_categorical(k, alpha):
    alpha0 = sum(alpha)
    res = special.beta(alpha0, 1) / special.beta(alpha[k], 1)
    return res


def Age2Comp(age, inv):  # This function is to specify the intensity component for the certain age(value) and certain transition index.  Interval looks like [ ).
    temp = 0
    while age >= inv[temp]:
        temp += 1
    return (temp)


def Predict_SR(Pars, lasts):
    """
    compute the average predictive probability given the number of tests
    :param Pars: estimated parameters
    :param lasts: estimated probability of the last state
    :return: average predictive probability for cytology, histology and HPV.
    """
    Alpha = Pars[0]
    cytology_pred_scores = []
    histology_pred_scores = []
    hpv_pred_scores = []

    indx =0
    for last, patient_observations, patient_tests in zip(lasts, observations_test, testTypes_test):
        indx += 1
        if indx % 1000 == 999:
            print("{}/{} individuals has been completed.".format(indx + 1, n_test))
        T_cyt = patient_tests[-1][0]
        T_hist = patient_tests[-1][1]
        T_hpv = patient_tests[-1][2]
        if T_cyt > 0:
            cyt_score = 0
            for S in range(4):
                cyt_score += ddirichlet_mutilnominal(x=patient_observations[-1][0, :], alpha=np.exp(Alpha[0])[S, :])*last[S]
            cytology_pred_scores.append(cyt_score)
        if T_hist > 0:
            hist_score = 0
            for S in range(4):
                hist_score += ddirichlet_mutilnominal(x=patient_observations[-1][1, :], alpha=np.exp(Alpha[1])[S, :])*last[S]
            histology_pred_scores.append(hist_score)
        if T_hpv > 0:
            hpv_score = 0
            for S in range(4):
                hpv_score += ddirichlet_mutilnominal(x=patient_observations[-1][2, :2], alpha=np.exp(Alpha[2])[S, :])*last[S]
            hpv_pred_scores.append(hpv_score)
    print(cytology_pred_scores)
    print(histology_pred_scores)
    print(hpv_pred_scores)
    cytology_pred_avgscore = np.mean(np.asarray(cytology_pred_scores))
    histology_pred_avgscore = np.mean(np.asarray(histology_pred_scores))
    hpv_pred_avgscore = np.mean(np.asarray(hpv_pred_scores))
    return cytology_pred_avgscore, histology_pred_avgscore, hpv_pred_avgscore



if __name__ == "__main__":
    #################################
    ####### Initialization ##########
    #################################
    Initialization()

    #################################
    ######## Load EM estimates ######
    #################################
    Load_EM_res(verbose=True)

    #################################
    ####### HHMM Prediction #########
    #################################
    # # Compute predictive distribution of last second state given model index z
    # Compute_pos_last2(currPars)
    # # Compute predictive distribution of last state
    # Compute_pos_last(currPars, verbose=True)

    ################################
    ######## Save Results ##########
    ################################
    # with open("../../res/EM_2400/prediction_LS.pickle", "wb") as res:
    #     pickle.dump(lasts, res)

    ############################
    #######Load Results ########
    ############################
    with open("../../res/EM_2400/hierarchical_prediction_LS.pickle", "rb") as res:
        lasts = pickle.load(res)


    ts = time.time()
    print(Predict_SR(currPars, lasts))
    print("prediction costs {}s".format(time.time() - ts))