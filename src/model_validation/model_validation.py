#!/usr/bin/env python2
'''
Kaplan Meier approach to assess the hierarchical hidden markov model
'''

import argparse
import os
import sys
sys.path.append('../')
sys.path.append('~/')
### Libraries ###
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
from scipy.linalg import expm

### test
import pdb
import logging

### Parallel Library
from mpi4py import MPI

def Initialization(verbose = False):
    global comm, rank
    global ages, testTypes, observations, treatment_indx, censor_ages, death_states
    global ind, nTests, inv, n_inv, MPmatrixs, out_path
    global P_model, Alpha, Eta, W, C
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print ('rank: ', rank)
    print ('size: ', size)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of the experiment", default="Model_Validation")
    parser.add_argument("--min_age", help="minimum age", type=int, default=16)
    parser.add_argument("--dataset", help="data specification: updated_data, updated_nonzero_data are available", default="updated_data")
    parser.add_argument("--age_inv", help="age interval sepecification", default="inv4")
    parser.add_argument("--n_patients_per_proc", help="number of patients per process", type=int, default=100)
    parser.add_argument("--test", help="boolean indicator for testing", action='store_true')

    args = parser.parse_args()
    if args.test:
        out_path = args.name + '_' + str(args.min_age) + '_' + args.dataset + '_' + args.age_inv + '_' + str(args.n_patients_per_proc*size) + '_test'
    else:
        out_path = args.name + '_' + str(args.min_age) + '_' + args.dataset + '_' + args.age_inv + '_' + str(args.n_patients_per_proc*size)
    min_age = args.min_age
    do_truncate_ages = True if min_age > 16 else False
    dataset = args.dataset
    inv_indx = args.age_inv
    if rank == 0:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        logging.basicConfig(level=logging.DEBUG, filename='model_validation.log')


    ##########################
    ##### Initialization #####
    ##########################
    nStates_0 = 2
    nStates_1 = 4
    nTests = 3

    # this number should be less than or equal to 100.
    n_patients_per_proc = args.n_patients_per_proc
    ind = ["Alpha", "Eta", "W", "C"]  # ind specifies which parameters need to be optimized:x ["Alpha", "W", "Gamma", "Zeta", "Eta", "A", "C"]
    inv_list = {"inv14": [20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 53, 57, 60, 200],
                "inv13": [20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 55, 60, 200],
                "inv12": [20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 60, 200],
                "inv11": [20, 23, 27, 30, 33, 37, 40, 45, 50, 60, 200],
                "inv10": [19, 22, 25, 30, 35, 40, 45, 50, 55, 200 ], # expert advice 
                "inv9": [23, 25, 30, 35, 40, 45, 50, 60, 200], # expert advice
                "inv8": [20, 25, 30, 35, 40, 50, 60, 200], # best AIC and Likelihood with 3000 procs.
                "inv7": [20, 25, 30, 40, 50, 60, 200], # best AIC with 300 procs
                "inv6": [23, 30, 40, 50, 60, 200],
                "inv5": [23, 35, 45, 60, 200],
                "inv4": [23, 30, 60, 200],
                "inv3": [29, 69, 200], # close second AIC with 300 procs
                "inv2": [23, 200],
                "inv1": [200]}
    inv = inv_list[inv_indx]
    n_inv = len(inv)
    
    data_location = ''
    if dataset == 'updated_data':
        data_location = '../../distributed_updated_data/'
    elif dataset == 'updated_nonzero_data':
        data_location = '../../distributed_updated_nonzero_data/'
    else:
        print("dataset {} is not available".format(dataset))

    subdata_location = data_location + 'p%s/'%(str(rank))

    subdata_location = data_location + 'p%s/'%(str(rank))

    # load data
    times           =  pickle.load(open( subdata_location + 'mcmcPatientTimes', 'r'))          #
    testTypes       =  pickle.load(open( subdata_location + 'mcmcPatientTestTypes', 'r'))      #
    observations    =  pickle.load(open( subdata_location + 'mcmcPatientObservations', 'r'))   #
    regressors      =  pickle.load(open( subdata_location + 'mcmcPatientRegressors', 'r'))     #
    treatment_indx  =  pickle.load(open( subdata_location + 'mcmcPatientTreatmentIndx', 'r'))           #
    censor_ages     =  pickle.load(open( subdata_location + 'mcmcPatientCensorDates', 'r'))             #
    death_states    =  pickle.load(open( subdata_location + 'mcmcPatientDeathStates', 'r'))             #


    # define Markov Process topology with MPmatrix. The diagonal should be zeros.
    # A one in element (i,j) indicates a possible transition between states i and j.
    MPmatrix_0 = np.zeros([nStates_0+1, nStates_0+1])
    MPmatrix_0[0,1] = MPmatrix_0[1,0] = MPmatrix_0[0,2] = MPmatrix_0[1,2] = 1
    MPmatrix_1 = np.zeros([nStates_1+1, nStates_1+1])
    MPmatrix_1[0,1] = MPmatrix_1[1,0] = MPmatrix_1[1,2] = MPmatrix_1[2,1] = MPmatrix_1[2,3] = MPmatrix_1[:-1,-1] = 1
    MPmatrixs = [MPmatrix_0, MPmatrix_1]

    temp_ages = regressors[1]
    ages = []
    # Reset age
    for temp_patient_ages, patient_times in zip(temp_ages, times):
        new_patient_ages = temp_patient_ages[0] + patient_times/12.0
        ages.append(new_patient_ages)

    nPatients = len(times)


    testTypes       =  testTypes[:n_patients_per_proc]
    observations    =  observations[:n_patients_per_proc]
    ages            =  ages[:n_patients_per_proc]
    treatment_indx  =  treatment_indx[:n_patients_per_proc]
    censor_ages     =  censor_ages[:n_patients_per_proc]
    death_states    =  death_states[:n_patients_per_proc]

    
    # If we want to drop patients younger than min_age > 16, do_truncate_ages should be set to True
    if do_truncate_ages:
        drop_patient_indx = []
        for p in xrange(nPatients):
            keep_obs = ages[p] >= min_age
            keep_indx = np.where(ages[p] >= min_age)[0]

            if len(keep_indx) > 1:
                first_indx = keep_indx[0]
                print 'keep_indx: ', keep_indx
                print 'first_indx: ', first_indx
                for ti in sorted(range(len(treatment_indx[p])), reverse=True):
                    print 'ti: ', ti
                    print 'treatment_indx[p][ti]: ', treatment_indx[p][ti]
                    if treatment_indx[p][ti] < first_indx:
                        del treatment_indx[p][ti]
                    else:
                         treatment_indx[p][ti] -= first_indx

                times[p]           =  times[p][keep_obs]
                testTypes[p]       =  testTypes[p][keep_obs,:]
                observations[p]    =  observations[p][keep_obs,:,:]
                ages[p]            =  ages[p][keep_obs]

            else:
                drop_patient_indx.append(p)

        if len(drop_patient_indx) > 0:
            for p in sorted(drop_patient_indx, reverse=True):
                del times[p]
                del testTypes[p]
                del observations[p]
                del ages[p]
                del treatment_indx[p]
                del censor_ages[p]
                del death_states[p]

    print 'original n patients: ', nPatients
    nPatients = len(ages)
    print 'Length of times: ', nPatients

    ### Upload parameters
    res_em = pickle.load(open('../EM_hierarchical/EM_hierarchical_16_updated_data_inv4_continuous_240000/res'))
    currZ_pos_list = res_em[1]
    currStates_list = res_em[2]
    Alpha = res_em[3]
    Eta = res_em[4]
    W = res_em[5]
    C = res_em[6]

    currPs = np.array([prob for currZ_pos in currZ_pos_list for prob in currZ_pos])
    P_model = (len(currPs[currPs > 0.5])+0.0) / len(currPs) ### Posterior probability of patient who belongs to model 1
 
    if verbose:
        print (P_model)
    return (0)

def ParList2ParVec(Alpha, Eta, W, C, ind): 
    parameter_vector = np.array([])
    
    if "Alpha" in ind:
        Alpha_vec = np.array([par for Alphai in Alpha for Alphaij in Alphai for par in Alphaij.reshape(-1)])
        parameter_vector = np.append(parameter_vector, Alpha_vec)
    if "Eta" in ind: 
        Eta_vec = np.array([par for Etai in Eta for par in Etai.reshape(-1)])
        parameter_vector = np.append(parameter_vector ,Eta_vec)
    if "W" in ind:
        W_vec = np.array([par for Wi in W for par in Wi.reshape(-1)])
        parameter_vector = np.append(parameter_vector, W_vec)
    if "C" in ind:
        C_vec = np.array([par for Ci in C for par in Ci.reshape(-1)])
        parameter_vector = np.append(parameter_vector, C_vec)
    return (parameter_vector)

def Age2Comp(age, inv): # This function is to specify the intensity component for the certain age(value) and certain transition index.  Interval looks like [ ).
    temp = 0
    while age >= inv[temp]:
        temp += 1
    return (temp)

def ProbTransition_interval(MPmatrix, dt, lambdas):
    '''
        'MPmatrix' should be a square N-by-N matrix of ones and zeros that defines the intensity matrix of the markov process.
        A 1 at element ij indicates a possible transition between states i and j.
        A 0 at element ij means no possible transition between states i and j.

        -- Because this is a continuous time Markov Process the diagonals are forced to be zero.
        -- 'lambdas' is an array of transition intensities for the given patient at a given time interval.
        -- dt is a scalar. It is the difference in time between two observations.

        '''
    matrix = np.array(MPmatrix,copy=True)
    matrix[np.where(matrix > 0)] = lambdas

    for i in xrange(matrix.shape[0]):
        matrix[i,i] = - np.sum(matrix[i,:])
    try:
        out = expm(dt*matrix)  # so far so good...
    except:
        try:
            out = expm2(dt*matrix)
        except:
            print 'lambdas: ', lambdas
            print 'matrix: ', matrix
            print 'dt: ', dt
            print 'dt*Lambda: ', dt*matrix
            print 'exp(dt*Lambda): ', expm2(dt*matrix)
            out  = expm2(dt*matrix)
            #out = np.eye(MPmatrix.shape[0])

    # Normalize the probablity matrix
    out = np.where(out < 0, 0., out)
    out = np.where(out > 1, 1., out)
    norm = np.repeat(np.sum(out,1),out.shape[0]).reshape(out.shape)
    out = out/norm
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
            temp_W = np.array([w[temp_component] for w in W])
            matrix = np.dot(matrix, ProbTransition_interval(MPmatrix, dt, np.exp(temp_W)))
            temp = inv[temp_component]
        else:
            dt = end - temp
            temp_W = np.array([w[temp_component] for w in W])
            matrix = np.dot(matrix, ProbTransition_interval(MPmatrix, dt, np.exp(temp_W)))
            temp = inv[temp_component]


    out = matrix
    #Normalize the probability matrix
    out = np.where(out < 0, 0., out)
    out = np.where(out > 1, 1., out)
    norm = np.repeat(np.sum(out,1),out.shape[0]).reshape(out.shape)
    out = out/norm
    return out

### Compute the patients' ages when they are first to come for the screening tests
def Comp_start(verbose = False):
    global total_first_ages
    first_ages = [patient_ages[0] for patient_ages in ages]
    ages_list = comm.gather(first_ages, 0)
    if rank == 0:
        total_first_ages = [first_age for first_ages in ages_list for first_age in first_ages]
        if verbose:
            # print total_first_ages
            plt.figure()
            plt.hist(total_first_ages, bins = 100)
            plt.show()
    return (0)

### Compute the empirical distribution of time intervals before observations
def Comp_ED_res(verbose = False):
    global ED_0, ED_1, ED_2, ED_3
    ED_times = [[] for i in range(4)]
    for patient_indx, patient_ages, patient_observations in zip(np.arange(len(ages)), ages, observations):
        for patient_time_stamp, patient_age, patient_observation in zip(np.arange(len(patient_ages)), patient_ages, patient_observations):
            if np.sum(patient_observation[:2]) == 0:
                break
            if patient_time_stamp == 0:
                curr_age = patient_age
                curr_observation = np.max(np.where(np.sum(patient_observation[:2], axis = 0)>0))
            else: 
                prev_age = curr_age
                prev_observation = curr_observation
                curr_age = patient_age
                curr_observation = np.max(np.where(np.sum(patient_observation[:2], axis = 0)>0))
                ED_times[prev_observation].append(curr_age - prev_age)
    ED_times_0_list = comm.gather(ED_times[0], 0)
    ED_times_1_list = comm.gather(ED_times[1], 0)
    ED_times_2_list = comm.gather(ED_times[2], 0)
    ED_times_3_list = comm.gather(ED_times[3], 0)
    if rank == 0:
        ED_0 = [time for ED_times_0 in ED_times_0_list for time in ED_times_0]
        ED_1 = [time for ED_times_1 in ED_times_1_list for time in ED_times_1]
        ED_2 = [time for ED_times_2 in ED_times_2_list for time in ED_times_2]
        ED_3 = [time for ED_times_3 in ED_times_3_list for time in ED_times_3]
        if verbose:
            plt.figure()
            plt.hist(ED_0, bins = 100)
            plt.title("Empirical distribution of time intervals from observation 0")
            plt.show()
            plt.hist(ED_1, bins = 100)
            plt.title("Empirical distribution of time intervals from observation 1")
            plt.show()
            plt.figure()
            plt.hist(ED_2, bins = 100)
            plt.title("Empirical distribution of time intervals from observation 2")
            plt.show()
            plt.figure()
            plt.hist(ED_3, bins = 100)
            plt.title("Empirical distribution of time intervals from observation 3")
            plt.show()
 
### Option with respect ot the definition of Kaplan Meier estimators
def Option_KM(patient_observation, KM_option):
    if KM_option == 0:  # failure measured from observed 0 to observed (1,2,3).  Ignore HPV.
        if np.sum(patient_observation[:2, 1:]) > 0:
            return 1
        else:
            return 0

    if KM_option == 1: # failure measured from observed (0,1) to observed (2,3).  Ignore HPV.
        if np.sum(patient_observation[:2, 2:]) > 0:
            return 1
        else:
            return 0

### Draw the empirical Kaplan Meier curve
def Draw_observations(verbose = False, KM_option = 0):
    global inv_x_observed, inv_y_observed
    maximum = 110.0
    fails = np.repeat(maximum, len(ages))
    indx = np.arange(0, len(ages))
    for patient_indx, patient_ages, patient_observations in zip(indx, ages, observations):
        patient_first_age = patient_ages[0]
        for patient_age, patient_observation in zip(patient_ages, patient_observations):
            if Option_KM(patient_observation, KM_option):
                fails[patient_indx] = patient_age - patient_first_age
                break
    fails_list = comm.gather(fails, 0)
    if rank == 0:
        total_fails = np.array([fail for fails in fails_list for fail in fails])
        total_fails.sort()
        # print (type(total_fails), total_fails.shape)
        probs = (1. - np.arange(0, len(total_fails)).astype(float)/len(total_fails))
        # print (probs)
        inv_x_observed = np.concatenate([np.array([0]), np.repeat(total_fails, 2)], axis = 0)
        inv_y_observed = np.concatenate([np.repeat(probs, 2), np.array([0])], axis = 0)
        if verbose:
            plt.figure()
            plt.plot(inv_x_observed, inv_y_observed)
            plt.show()
    return (0)


def Draw_simulations(n_simulate = 1000, verbose = False, option = 2, KM_option = 0):
    maximum = 110.0
    fails = np.repeat(maximum, n_simulate)
    if KM_option == 0:
        threshold = 1
    if KM_option == 1:
        threshold = 2
    for i in range(n_simulate):
        # print (i)
        # First stage
        if option == 2:
            patient_z = np.random.choice(np.array([0,1]), p = np.array([1- P_model, P_model]))
        if option == 0:
            patient_z = 0
        if option == 1:
            patient_z = 1

        # print "patient_model", patient_z  #####
        
        patient_first_age = patient_age = np.random.choice(total_first_ages)
        
        # print "patient_age", patient_age #####
        
        # patinet_ages = [patient_age]
        if patient_z == 0:
            patient_state = np.random.choice(np.arange(2), p = np.random.dirichlet(np.exp(C[patient_z][:, Age2Comp(patient_age, inv)])))
        else:
            patient_state = np.random.choice(np.arange(4), p = np.random.dirichlet(np.exp(C[patient_z][:, Age2Comp(patient_age, inv)])))
        
        # print "patient_state", patient_state #####
        
        # patient_states = [patient_state] 
        # patient_tests
        patient_test = np.array([np.random.poisson(lam = np.exp(Eta[patient_z][patient_state, 0])), np.random.poisson(lam = np.exp(Eta[patient_z][patient_state, 1])), np.random.poisson(lam = np.exp(Eta[patient_z][patient_state, 2]))])
        if sum(patient_test[:2]) == 0:
            patient_test = np.array([1, 0, 0])

        # print "patient_test", patient_test

        # patient_tests = [patient_test]
        # Only simulate cytology and histology
        patient_observation = np.zeros([2, 4])
        if patient_test[0] > 0:
            patient_observation[0,:] = np.random.multinomial(n = patient_test[0], pvals = np.random.dirichlet(np.exp(Alpha[patient_z][0][patient_state]))) 
        if patient_test[1] > 0:
            patient_observation[1,:] = np.random.multinomial(n = patient_test[1], pvals = np.random.dirichlet(np.exp(Alpha[patient_z][1][patient_state]))) 
        # patient_observations = [patient_observation]
        patient_result = np.max(np.where(np.sum(patient_observation, axis = 0)>0)) 

        # print "patient_observation", patient_observation


        # Recursive
        while patient_result < threshold and patient_age < 110:
            # Update patient age
            prev_patient_age = patient_age
            if patient_result == 0:
                patient_age += np.random.choice(ED_0)
            if patient_result == 1:
                patient_age += np.random.choice(ED_1)

            # print "patient_age", patient_age

            # Update patient state
            p_transition = ProbTransition(MPmatrixs[patient_z], W[patient_z], prev_patient_age, patient_age, inv)
            if patient_z == 0:
                patient_state = np.random.choice(np.arange(3), p = p_transition[patient_state,:])
                if patient_state == 2:
                    break
            if patient_z == 1:
                patient_state = np.random.choice(np.arange(5), p = p_transition[patient_state,:])
                if patient_state == 4:
                    break

            # print "patient_state", patient_state 

            # Update patient_tests
            # patient_tests
            patient_test = np.array([np.random.poisson(lam = np.exp(Eta[patient_z][patient_state, 0])), np.random.poisson(lam = np.exp(Eta[patient_z][patient_state, 1])), np.random.poisson(lam = np.exp(Eta[patient_z][patient_state, 2]))])
            if sum(patient_test[:2]) == 0:
                patient_test = np.array([1, 0, 0]) 
            
            # print np.exp(Eta[patient_z][patient_state, :])
            # print "patient_test", patient_test

            # Update patient observation
            patient_observation = np.zeros([2, 4])
            if patient_test[0] > 0:
                patient_observation[0,:] = np.random.multinomial(n = patient_test[0], pvals = np.random.dirichlet(np.exp(Alpha[patient_z][0][patient_state]))) 
            if patient_test[1] > 0:
                patient_observation[1,:] = np.random.multinomial(n = patient_test[1], pvals = np.random.dirichlet(np.exp(Alpha[patient_z][1][patient_state]))) 
            # patient_observations = [patient_observation]
            patient_result = np.max(np.where(np.sum(patient_observation, axis = 0)>0))
            # print "patient_observation", patient_observation

        fails[i] = patient_age - patient_first_age
        
    # print "fails", fails
    fails.sort()
    probs = 1. - np.arange(0, n_simulate).astype(float)/len(fails)
    # inv_x = np.concatenate([np.array([0]), np.repeat(fails, 2)], axis = 0) ### standard KM piecewise constant function
    # inv_y = np.concatenate([np.repeat(probs, 2), np.array([0])], axis = 0) ### standard KM piecewise constant function
    inv_x = np.concatenate([np.array([0]), fails], axis = 0) ### standard KM piecewise linear function
    inv_y = np.concatenate([probs, np.array([0])], axis = 0) ### standard KM piecewise linear function

    if verbose:
        print("options: {}".format(option))
        plt.figure()
        plt.plot(inv_x, inv_y)
        plt.show()
    return inv_x, inv_y


def Draw_simulations_original(n_simulate = 1000, verbose = False, KM_option = 0):
    res_em_single = pickle.load(open('../EM_hierarchical/EM_16_updated_data_inv4_continuous_240000/res'))
    Alpha = res_em_single[2]
    Eta = res_em_single[3]
    W = res_em_single[4]
    C = res_em_single[5]
    MPmatrix = MPmatrixs[1]

    # print (Alpha, Eta, W, C)

    maximum = 110.0
    fails = np.repeat(maximum, n_simulate)
    if KM_option == 0:
        threshold = 1
    if KM_option == 1:
        threshold = 2
    for i in range(n_simulate):
        # print (i)
        # First stage
        
        patient_first_age = patient_age = np.random.choice(total_first_ages)
        
        # print "patient_age", patient_age #####
        
        # patinet_ages = [patient_age]
        patient_state = np.random.choice(np.arange(4), p = np.random.dirichlet(np.exp(C[:, Age2Comp(patient_age, inv)])))
        
        # print "patient_state", patient_state #####
        
        # patient_states = [patient_state] 
        # patient_tests
        patient_test = np.array([np.random.poisson(lam = np.exp(Eta[patient_state, 0])), np.random.poisson(lam = np.exp(Eta[patient_state, 1])), np.random.poisson(lam = np.exp(Eta[patient_state, 2]))])
        if sum(patient_test[:2]) == 0:
            patient_test = np.array([1, 0, 0])

        # print "patient_test", patient_test

        # patient_tests = [patient_test]
        # Only simulate cytology and histology
        patient_observation = np.zeros([2, 4])
        if patient_test[0] > 0:
            patient_observation[0,:] = np.random.multinomial(n = patient_test[0], pvals = np.random.dirichlet(np.exp(Alpha[0][patient_state]))) 
        if patient_test[1] > 0:
            patient_observation[1,:] = np.random.multinomial(n = patient_test[1], pvals = np.random.dirichlet(np.exp(Alpha[1][patient_state]))) 
        # patient_observations = [patient_observation]
        patient_result = np.max(np.where(np.sum(patient_observation, axis = 0)>0)) 

        # print "patient_observation", patient_observation


        # Recursive
        while patient_result < threshold and patient_age < 110:
            # Update patient age
            prev_patient_age = patient_age
            if patient_result == 0:
                patient_age += np.random.choice(ED_0)
            if patient_result == 1:
                patient_age += np.random.choice(ED_1)

            # print "patient_age", patient_age

            # Update patient state
            p_transition = ProbTransition(MPmatrix, W, prev_patient_age, patient_age, inv)
            
            patient_state = np.random.choice(np.arange(5), p = p_transition[patient_state,:])
            if patient_state == 4:
                break

            # print "patient_state", patient_state 

            # Update patient_tests
            # patient_tests
            patient_test = np.array([np.random.poisson(lam = np.exp(Eta[patient_state, 0])), np.random.poisson(lam = np.exp(Eta[patient_state, 1])), np.random.poisson(lam = np.exp(Eta[patient_state, 2]))])
            if sum(patient_test[:2]) == 0:
                patient_test = np.array([1, 0, 0]) 
            
            # print np.exp(Eta[patient_z][patient_state, :])
            # print "patient_test", patient_test

            # Update patient observation
            patient_observation = np.zeros([2, 4])
            if patient_test[0] > 0:
                patient_observation[0,:] = np.random.multinomial(n = patient_test[0], pvals = np.random.dirichlet(np.exp(Alpha[0][patient_state]))) 
            if patient_test[1] > 0:
                patient_observation[1,:] = np.random.multinomial(n = patient_test[1], pvals = np.random.dirichlet(np.exp(Alpha[1][patient_state]))) 
            # patient_observations = [patient_observation]
            patient_result = np.max(np.where(np.sum(patient_observation, axis = 0)>0))
            

            # print "patient_observation", patient_observation

        fails[i] = patient_age - patient_first_age
        
    # print "fails", fails
    fails.sort()
    probs = 1. - np.arange(0, n_simulate).astype(float)/len(fails)
    # inv_x = np.concatenate([np.array([0]), np.repeat(fails, 2)], axis = 0) ### standard KM piecewise constant function
    # inv_y = np.concatenate([np.repeat(probs, 2), np.array([0])], axis = 0) ### standard KM piecewise constant function
    inv_x = np.concatenate([np.array([0]), fails], axis = 0) ### standard KM piecewise linear function
    inv_y = np.concatenate([probs, np.array([0])], axis = 0) ### standard KM piecewise linear function

    if verbose:
        print("Single hmm")
        plt.figure()
        plt.plot(inv_x, inv_y)
        plt.show()
    return (inv_x, inv_y)

# def Draw_simulations_original(n_simulate = 10000, verbose = False, KM_option = 0):
#     res = pickle.load(open('em_state_pars_lbfgsb_1200procs_noZeta_noGamma_10patients_16_zero_inv4'))
#     Alpha = res[2]
#     Eta = res[3]
#     W = res[4]
#     A = res[7]
#     C = np.array(res[9])

#     maximum = 110.0
#     fails = np.repeat(maximum, n_simulate)
#     if KM_option == 0:
#         threshold = 1
#     if KM_option == 1:
#         threshold = 2

#     for i in range(n_simulate):
#         # First stage

#         patient_first_age = patient_age = np.random.choice(total_first_ages)
        
#         # print "patient_age", patient_age #####
        
#         # patinet_ages = [patient_age]
#         patient_state = np.random.choice(np.arange(4), p = np.exp(C[:, Age2Comp(patient_age, inv)])/np.sum(np.exp(C[:, Age2Comp(patient_age, inv)])))
        
#         # print "patient_state", patient_state #####
        
#         # patient_states = [patient_state] 
#         # patient_tests
#         patient_test = np.array([np.random.poisson(lam = Eta[patient_state, 0]), np.random.poisson(lam = Eta[patient_state, 1]), np.random.poisson(lam = Eta[patient_state, 2])])
#         if sum(patient_test[:2]) == 0:
#             patient_test = np.array([1, 0, 0])

#         # print "patient_test", patient_test

#         # patient_tests = [patient_test]
#         # Only simulate cytology and histology
#         patient_observation = np.zeros([2, 4])
#         if patient_test[0] > 0: # Simulate cytoplogy screening test results
#             patient_observation[0,:] = np.random.multinomial(n = patient_test[0], pvals = np.exp(A[patient_state, :])/np.sum(np.exp(A[patient_state, :]))) 
#         if patient_test[1] > 0: # Simulate histology screening test results
#             patient_observation[1,:] = np.random.multinomial(n = patient_test[1], pvals = Alpha[0][patient_state, :]) 
#         # patient_observations = [patient_observation]
#         patient_result = np.max(np.where(np.sum(patient_observation, axis = 0)>0)) 

#         # print "patient_observation", patient_observation

#         # Recursive
#         while patient_result < threshold and patient_age < 110:
#             # Update patient age
#             prev_patient_age = patient_age
#             if patient_result == 0:
#                 patient_age += np.random.choice(ED_0)
#             if patient_result == 1:
#                 patient_age += np.random.choice(ED_1)

#             # print "patient_age", patient_age

#             # Update patient state
#             p_transition = ProbTransition(MPmatrixs[1], W, prev_patient_age, patient_age, inv)    
#             patient_state = np.random.choice(np.arange(5), p = p_transition[patient_state,:])
#             if patient_state == 4:
#                 break

#             # print "patient_state", patient_state 

#             # Update patient_tests
#             # patient_tests
#             patient_test = np.array([np.random.poisson(lam = Eta[patient_state, 0]), np.random.poisson(lam = Eta[patient_state, 1]), np.random.poisson(lam = Eta[patient_state, 2])])
#             if sum(patient_test[:2]) == 0:
#                 patient_test = np.array([1, 0, 0])
            
#             # print np.exp(Eta[patient_z][patient_state, :])
#             # print "patient_test", patient_test

#             # Update patient observation
#             patient_observation = np.zeros([2, 4])
#             if patient_test[0] > 0: # Simulate cytoplogy screening test results
#                 patient_observation[0,:] = np.random.multinomial(n = patient_test[0], pvals = np.exp(A[patient_state, :])/np.sum(np.exp(A[patient_state, :]))) 
#             if patient_test[1] > 0: # Simulate histology screening test results
#                 patient_observation[1,:] = np.random.multinomial(n = patient_test[1], pvals = Alpha[0][patient_state, :]) 
#             # patient_observations = [patient_observation]
#             patient_result = np.max(np.where(np.sum(patient_observation, axis = 0)>0))
            

#             # print "patient_observation", patient_observation

#         fails[i] = patient_age - patient_first_age
        
#     # print "fails", fails
#     fails.sort()
#     probs = 1. - np.arange(0, n_simulate).astype(float)/len(fails)
#     inv_x = np.concatenate([np.array([0]), np.repeat(fails, 2)], axis = 0)
#     inv_y = np.concatenate(
#         [np.repeat(probs, 2), np.array([0])], axis = 0)
    
#     if verbose:
#         plt.figure()
#         plt.plot(inv_x, inv_y)
#         plt.show()
#     return (inv_x, inv_y)

def Interpolation(inv_x, inv_y, x):
    return (np.interp(x, inv_x, inv_y))

def Kaplan_Meier(verbose = False, KM_option = 0, N_sim = 100, n_sim = 1000):
    # plot the observed data
    Draw_observations(verbose = False, KM_option = KM_option)
    # pdb.set_trace()
    if rank == 0:
        x_mixed_list = []
        y_mixed_list = []
        x_original_list = []
        y_original_list = []
        x_grids = np.linspace(0, 30, 301)
        for i in range(N_sim):
            t_s = time.time()
            logging.info("{}th starts".format(i))
            inv_x_mixed, inv_y_mixed = Draw_simulations(n_simulate = n_sim, verbose = False, option = 2, KM_option = KM_option)
            inv_x_original, inv_y_original = Draw_simulations_original(n_simulate = n_sim, verbose = False, KM_option = KM_option)
            y_mixed = Interpolation(inv_x_mixed, inv_y_mixed, x_grids)
            y_original = Interpolation(inv_x_original, inv_y_original, x_grids)
            # inv_x_0, inv_y_0 = Draw_simulations(verbose = False, option = 0, KM_option = KM_option)
            # inv_x_1, inv_y_1 = Draw_simulations(verbose = False, option = 1, KM_option = KM_option)
            y_mixed_list.append(y_mixed)
            y_original_list.append(y_original)
            print("{}th simulation costs {}s".format(i, time.time()- t_s))
            logging.info("{}th simulation costs {}s".format(i, time.time()- t_s))
        y_mixed_mat = np.stack(y_mixed_list, axis = 0)
        y_original_mat = np.stack(y_original_list, axis = 0)
        y_mixed_quantiles = np.percentile(y_mixed_mat, [2.5, 50, 97.5], axis=0)
        y_original_quantiles = np.percentile(y_original_mat, [2.5, 50, 97.5], axis=0)

        # print(x_grids.shape, y_mixed_quantiles.shape, y_original_quantiles.shape)
        # pdb.set_trace()
        if verbose:
            plt.figure()
            plt.ylim(0, 1)
            plt.xlim(0, 30)
            plt.plot(inv_x_observed, inv_y_observed, color = 'k')
            # plot original curves
            plt.plot(x_grids, y_original_quantiles[1,:], color = 'b')
            plt.plot(x_grids[:,np.newaxis], y_original_quantiles[[0,2],:].T, color = 'b', linestyle='--')
            # plot mixed curves
            plt.plot(x_grids, y_mixed_quantiles[1,:], color = 'r')
            plt.plot(x_grids[:,np.newaxis], y_mixed_quantiles[[0,2],:].T, color = 'r', linestyle='--')
            # plt.legend(["Obeserved KM curve", "KM curve from the simple HMM", "KM curve from the hierarchical HMM"], loc=3, fontsize = 'x-small')        
            # plt.show()
            if KM_option == 0:
                plt.savefig("KM_curve_shared_version0.png")
            if KM_option == 1:
                plt.savefig("KM_curve_shared_version1.png")
        
        if KM_option == 0:
            with open(out_path + "/model_validation_version0" , "wb") as em_res:
                # pickle.dump([inv_x_observed, inv_y_observed, inv_x_original, inv_y_original, inv_x_mixed, inv_y_mixed, inv_x_0, inv_y_0, inv_x_1, inv_y_1], em_res)
                pickle.dump([inv_x_observed, inv_y_observed, x_grids, y_original_mat, y_mixed_mat], em_res)
        if KM_option == 1:
            with open(out_path + "/model_validation_version1", "wb") as em_res:
                # pickle.dump([inv_x_observed, inv_y_observed, inv_x_original, inv_y_original, inv_x_mixed, inv_y_mixed, inv_x_0, inv_y_0, inv_x_1, inv_y_1], em_res)
                pickle.dump([inv_x_observed, inv_y_observed, x_grids, y_original_mat, y_mixed_mat], em_res)
    return (0)  

if __name__ == "__main__":
    ######################
    ### Initialization ###
    ######################
    Initialization()
    # pdb.set_trace()
    if rank == 0:
        logging.info("Initialization complete")

    ######################################################################
    ### Compute the Empirical distribution of first screening test time ###
    ######################################################################
    Comp_start()
    if rank == 0:
        logging.info("Procedure 1 complete")

    # pdb.set_trace()
    Comp_ED_res()
    if rank == 0:
        logging.info("Procedure 2 complete")

    # pdb.set_trace()

    #####################
    ### Kalplan_Meier ###
    #####################
    # if rank == 0:
    #     print ("Start to plot Kaplan Meier curve for version 0")
    #     logging.info("Start to plot Kaplan Meier curve for version 0")
    # Kaplan_Meier(True, KM_option = 0, N_sim = 100, n_sim = 100)
    if rank == 0:
        print ("Start to plot Kaplan Meier curve for version 1")
        logging.info("Start to plot Kaplan Meier curve for version 1")
    Kaplan_Meier(True, KM_option = 1, N_sim = 100, n_sim = 100)

    ### Divided by ages groups ###
    