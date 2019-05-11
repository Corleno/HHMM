#!/usr/bin/env python2
'''

EM Algorithm for DBN model.
Most recent version as of 12/28/17.

'''

import argparse
import os
# import logging
import sys
sys.path.append('../')
sys.path.append('~/')
### Libraries ###
import pickle
import time
import autograd.numpy as np
from scipy import optimize
from autograd.scipy import stats
from autograd.scipy import special
from autograd import grad

### Local Library
from matfuncs import expm

### Parallel Library
from mpi4py import MPI

def Initialization():
    global comm, rank
    global ages, testTypes, observations, treatment_indx, censor_ages, death_states, ind, nTests, inv, n_inv, MPmatrixs, nPatients
    global out_path, max_steps_em, max_steps_optim, model, autograd_optim, p, args
    global currPars, curr_parameter_vector
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('rank: ', rank)
    print('size: ', size)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of the experiment", default="EM_hierarchical")
    parser.add_argument("--min_age", help="minimum age", type=int, default=16)
    parser.add_argument("--dataset", help="data specification: updated_data, updated_nonzero_data, survey_patient_data are available", default="updated_data")
    parser.add_argument("--age_inv", help="age interval sepecification", default="inv4")
    parser.add_argument("--n_patients_per_proc", help="number of patients per process", type=int, default=100)
    parser.add_argument("--max_steps_em", help="maximum number of EM iterations", type=int, default=100)
    parser.add_argument("--max_steps_optim", help="maximum number of optim iterations", type=int, default=5)
    parser.add_argument("--model", help="discrete or continuous model", default="continuous")
    parser.add_argument("--test", help="boolean indicator for testing", action='store_true')
    parser.add_argument("--autograd_optim", help="boolean indicator for auto-gradient in optimization", action='store_true')
    parser.add_argument("--Z_prior", help="prior probability of Model 1", type=np.float32, default=0.5)
    parser.add_argument("--bootstrap_seed", help="seed number setting in Boostrap", type=int, default=0)
    parser.add_argument("--bootstrap_total", help="number of total seeds in pool", type = int, default=5000)

    args = parser.parse_args()

    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass

    # if np.isinf(args.n_patients_per_proc):
    #     total_num_patient_str = 'all'
    # else:
    #     total_num_patient_str = str(args.n_patients_per_proc*size)
    total_num_patient_str = str(args.n_patients_per_proc*size)
    if args.test:
        out_folder = args.name + '_' + str(args.min_age) + '_' + args.dataset + '_' + args.age_inv + '_' + args.model + '_' + total_num_patient_str + '_test'
    else:
        out_folder = args.name + '_' + str(args.min_age) + '_' + args.dataset + '_' + args.age_inv + '_' + args.model + '_' + total_num_patient_str
    
    min_age = args.min_age
    do_truncate_ages = True if min_age > 16 else False
    dataset = args.dataset
    inv_indx = args.age_inv
    max_steps_em = args.max_steps_em 
    max_steps_optim = args.max_steps_optim
    model = args.model
    autograd_optim = args.autograd_optim
    p = args.Z_prior 
    # indx for rank where we assume totally we have 5000 folders of data.
    np.random.seed(args.bootstrap_seed)
    rank_dict = np.random.choice(args.bootstrap_total, size).astype(np.int32)
    
    # logging.basicConfig(filename=out_path + "/em.log", level = logging.INFO, format='%(asctime)s %(message)s')       

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
    
    if dataset == 'updated_data':
        data_location = '../../distributed_updated_data/'
    elif dataset == 'updated_nonzero_data':
        data_location = '../../distributed_updated_nonzero_data/'
    elif dataset == 'survey_patient_data':
        data_location = '../../survey_patient_data/'
    else:
        print("dataset {} is not available".format(dataset))


    subdata_location = data_location + 'p%s/'%(str(rank_dict[rank]))

    # load data
    times           =  pickle.load(open( subdata_location + 'mcmcPatientTimes', 'r'))          #
    testTypes       =  pickle.load(open( subdata_location + 'mcmcPatientTestTypes', 'r'))      #
    observations    =  pickle.load(open( subdata_location + 'mcmcPatientObservations', 'r'))   #
    regressors      =  pickle.load(open( subdata_location + 'mcmcPatientRegressors', 'r'))     #
    treatment_indx  =  pickle.load(open( subdata_location + 'mcmcPatientTreatmentIndx', 'r'))           #
    censor_ages     =  pickle.load(open( subdata_location + 'mcmcPatientCensorDates', 'r'))             #
    death_states    =  pickle.load(open( subdata_location + 'mcmcPatientDeathStates', 'r'))             #

    # Move to outcome dir
    if not os.path.exists("outcome"):
        os.makedirs("outcome")
    os.chdir("outcome")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    os.chdir(out_folder)

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

    n_patients_per_proc = min(n_patients_per_proc, nPatients)
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

    # ### Initialize parameters
    # currAlpha_0 = [np.zeros([nStates_0,4]), np.zeros([nStates_0,4]), np.zeros([nStates_0, 2])]
    # currAlpha_1 = [np.zeros([nStates_1,4]), np.zeros([nStates_1,4]), np.zeros([nStates_1, 2])]
    # currAlpha = [currAlpha_0, currAlpha_1]
    # currEta_0 = np.zeros([nStates_0,3])
    # currEta_1 = np.zeros([nStates_1,3])
    # currEta = [currEta_0, currEta_1]
    # currW_0 = np.zeros([4, n_inv])
    # currW_1 = np.zeros([9, n_inv])
    # currW = [currW_0, currW_1]
    # currC_0 = np.zeros([nStates_0, n_inv])
    # currC_1 = np.zeros([nStates_1, n_inv])
    # currC = [currC_0, currC_1]

    ### Set informative initial parameters
    temp = 4
    currAlpha_0 = [np.zeros([nStates_0,4]), np.zeros([nStates_0,4]), np.zeros([nStates_0, 2])]
    currAlpha_0[0][0, 0] = currAlpha_0[0][1,1] = temp
    currAlpha_0[1][0, 0] = currAlpha_0[1][1,1] = temp
    currAlpha_1 = [np.zeros([nStates_1,4]), np.zeros([nStates_1,4]), np.zeros([nStates_1, 2])]
    currAlpha_1[0][0,0] = currAlpha_1[0][1,1] = currAlpha_1[0][2,2] = currAlpha_1[0][3,3] = temp
    currAlpha_1[1][0,0] = currAlpha_1[1][1,1] = currAlpha_1[1][2,2] = currAlpha_1[1][3,3] = temp
    currAlpha_1[2][3,0] = -2
    currAlpha_1[2][3,1] = 2
    currAlpha = [currAlpha_0, currAlpha_1]
    currEta_0 = np.zeros([nStates_0,3])
    currEta_1 = np.zeros([nStates_1,3])
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
        currW_0 = -4*np.ones([4, n_inv])
        currW_1 = -4*np.ones([9, n_inv])
    currW = [currW_0, currW_1]
    currC_0 = np.zeros([nStates_0, n_inv])
    currC_1 = np.zeros([nStates_1, n_inv])
    currC = [currC_0, currC_1]    

    currPars = [currAlpha, currEta, currW, currC]
    curr_parameter_vector = ParList2ParVec(currAlpha, currEta, currW, currC, ind)
    # currAlpha, currEta, currW, currC = ParVec2ParList(curr_parameter_vector, ind)
    # print (currAlpha, currEta, currW, currC)

    return 0

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

def ParList2ParVec_group(Alpha, Eta, W, C, ind):
    parameter_vector = np.array([])
    
    if "Alpha" in ind:
        Alpha_vec = np.array([par for Alphaj in Alpha for par in Alphaj.reshape(-1)])
        parameter_vector = np.append(parameter_vector, Alpha_vec)
    if "Eta" in ind: 
        Eta_vec = np.array([par for par in Eta.reshape(-1)])
        parameter_vector = np.append(parameter_vector ,Eta_vec)
    if "W" in ind:
        W_vec = np.array([par for par in W.reshape(-1)])
        parameter_vector = np.append(parameter_vector, W_vec)
    if "C" in ind:
        C_vec = np.array([par for par in C.reshape(-1)])
        parameter_vector = np.append(parameter_vector, C_vec)
    return (parameter_vector)

def ParVec2ParList(parameter_vector, n_inv, ind):
    s = 0
    if "Alpha" in ind:
        Alpha_vec = parameter_vector[s: s+60]
        s += 60
        Alpha0 = Alpha_vec[:20]
        Alpha1 = Alpha_vec[-40:]
        Alpha = [[Alpha0[0:8].reshape(2,4), Alpha0[8:16].reshape(2,4), Alpha0[16:20].reshape(2,2)], [Alpha1[:16].reshape(4,4), Alpha1[16:32].reshape(4,4), Alpha1[32:40].reshape(4,2)]]
    else:
        Alpha = currPars[0]
    if "Eta" in ind:
        Eta_vec = parameter_vector[s:s+18]
        s += 18
        Eta0 = Eta_vec[:6]
        Eta1 = Eta_vec[-12:]
        Eta = [Eta0.reshape(2,3), Eta1.reshape(4,3)]
    else:
        Eta = currPars[1]
    if "W" in ind:
        W_vec = parameter_vector[s: s+13*n_inv]
        s += 13 * n_inv
        W0 = W_vec[:4*n_inv]
        W1 = W_vec[-9*n_inv:]
        W = [W0.reshape(4, n_inv), W1.reshape(9, n_inv)]
    else:
        W = currPars[2]
    if "C" in ind:
        C_vec = parameter_vector[s: s+6*n_inv]
        s += 6*n_inv
        C0 = C_vec[:2*n_inv]
        C1 = C_vec[-4*n_inv:]
        C = [C0.reshape(2, n_inv), C1.reshape(4, n_inv)]
    else:
        C = currPars[3]
    return Alpha, Eta, W, C

def ParVecShared2ParVec(parameter_vector_shared, n_inv, ind):
    parameter_vector = np.array([])
    s = 0
    if "Alpha" in ind:
        Alpha_vec_shared = parameter_vector_shared[s: s+40]
        s += 40
        Alpha1 = Alpha_vec_shared
        Alpha0 = np.concatenate([Alpha1[:8], Alpha1[16:24], Alpha1[32:36]])
        Alpha_vec = np.append(Alpha0, Alpha1)
        parameter_vector = np.append(parameter_vector, Alpha_vec)
    if "Eta" in ind:
        Eta_vec_shared = parameter_vector_shared[s: s+12]
        s += 12
        Eta1 = Eta_vec_shared
        Eta0 = Eta1[:6]
        Eta_vec = np.append(Eta0, Eta1)
        parameter_vector = np.append(parameter_vector, Eta_vec)
    if "W" in ind:
        W_vec = parameter_vector_shared[s: s+13*n_inv]
        s += 13 * n_inv
        parameter_vector = np.append(parameter_vector, W_vec)
    if "C" in ind:
        C_vec = parameter_vector_shared[s: s+6*n_inv]
        s += 6 * n_inv
        parameter_vector = np.append(parameter_vector, C_vec)

    return parameter_vector

def ParVec2ParVecShared(parameter_vector, n_inv, ind):
    parameter_vector_shared = np.array([])
    s = 0
    if "Alpha" in ind:
        Alpha_vec = parameter_vector[s: s+60]
        s += 60
        Alpha0 = Alpha_vec[:20]
        Alpha1 = Alpha_vec[-40:]
        Alpha_vec_shared = Alpha1
        parameter_vector_shared = np.append(parameter_vector_shared, Alpha_vec_shared)
    if "Eta" in ind:
        Eta_vec = parameter_vector[s: s+18]
        s += 18
        Eta0 = Eta_vec[:6]
        Eta1 = Eta_vec[-12:]
        Eta_vec_shared = Eta1
        parameter_vector_shared = np.append(parameter_vector_shared, Eta_vec_shared)
    if "W" in ind:
        W_vec = parameter_vector[s: s+13*n_inv]
        s += 13 * n_inv
        parameter_vector_shared = np.append(parameter_vector_shared, W_vec)
    if "C" in ind:
        C_vec = parameter_vector[s: s+6*n_inv]
        s += 6 * n_inv
        parameter_vector_shared = np.append(parameter_vector_shared, C_vec)

    return parameter_vector_shared

def ParVec2ParList_group(Z, parameter_vector, n_inv, ind):
    s = 0
    if "Alpha" in ind:
        if Z == 0:
            n_Alpha = 20
        else:
            n_Alpha = 40
        Alpha_vec = parameter_vector[s: s+n_Alpha]
        s += n_Alpha
        if Z == 0:
            Alpha = [Alpha_vec[0:8].reshape(2,4), Alpha_vec[8:16].reshape(2,4), Alpha_vec[16:20].reshape(2,2)]
        else:
            Alpha = [Alpha_vec[:16].reshape(4,4), Alpha_vec[16:32].reshape(4,4), Alpha_vec[32:40].reshape(4,2)]
    if "Eta" in ind:
        if Z == 0:
            n_states = 2
        else:
            n_states = 4
        Eta_vec = parameter_vector[s:s+n_states*3]
        s += n_states*3
        Eta = Eta_vec.reshape(n_states,3)
    if "W" in ind:
        if Z == 0:
            n_tran = 4
        else:
            n_tran = 9
        W_vec = parameter_vector[s: s+n_tran*n_inv]
        s += n_tran * n_inv
        W = W_vec.reshape(n_tran, n_inv)
    if "C" in ind:
        if Z == 0:
            n_states = 2
        else:
            n_states = 4
        C_vec = parameter_vector[s: s+n_states*n_inv]
        s += n_states*n_inv
        C = C_vec.reshape(n_states, n_inv)
    return Alpha, Eta, W, C

def ddirichlet_mutilnominal(x, alpha):
    n = np.sum(x)
    alpha0 = sum(alpha)
    if n == 0:
        return 1
    else:
        return n*special.beta(alpha0,n)/np.prod(np.array([special.beta(alphak, xk)*xk for alphak, xk in zip(alpha, x) if xk > 0]))

def ddirichlet_categorical(k, alpha):
    alpha0 = sum(alpha)
    res = special.beta(alpha0, 1)/special.beta(alpha[k], 1)
    return res

def Age2Comp(age, inv): # This function is to specify the intensity component for the certain age(value) and certain transition index.  Interval looks like [ ).
    temp = 0
    while age >= inv[temp]:
        temp += 1
    return (temp)

def Softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def ProbTransition_interval(MPmatrix, dt, W):
    '''
        'MPmatrix' should be a square N-by-N matrix of ones and zeros that defines the intensity matrix of the markov process.
        A 1 at element ij indicates a possible transition between states i and j.
        A 0 at element ij means no possible transition between states i and j.

        -- Because this is a continuous time Markov Process the diagonals are forced to be zero.
        -- 'lambdas' is an array of transition intensities for the given patient at a given time interval.
        -- dt is a scalar. It is the difference in time between two observations.

        '''
    matrix = np.array(MPmatrix,copy=True)
    
    if model == 'continuous':
        if autograd_optim:
            matrix_filled = []
            indx = 0
            indx_row = -1
            for row in matrix:
                temp_row = []
                indx_row += 1
                for col in row:
                    if col > 0:
                        temp_row.append(np.exp(W)[indx])
                        indx += 1
                    else:
                        temp_row.append(0)
                temp_row[indx_row] = -sum(temp_row)
                matrix_filled.append(temp_row)
            out = expm(dt*np.array(matrix_filled))
        else:
            matrix_filled = np.zeros_like(matrix, dtype=np.float32)
            matrix_filled[np.where(matrix > 0)] = np.exp(W)
            for i in xrange(matrix.shape[0]):
                matrix_filled[i,i] = - np.sum(matrix_filled[i,:])
            out = expm(dt*matrix_filled)  # so far so good...
    elif model == 'discrete':
        if autograd_optim:
            n_dim = matrix.shape[0]
            np.fill_diagonal(matrix, -1)
            matrix_filled = []
            indx = 0
            indx_row = -1
            for row in matrix:
                temp_row = []
                temp_row0 = []
                indx_row0 = 0
                indx_row += 1
                for col in row:
                    if col == 1:
                        temp_row0.append(W[indx])
                        indx += 1
                    if col == -1:
                        temp_row0.append(0)
                temp_row0 = Softmax(np.array(temp_row0))
                # print (row, W, temp_row0)
                for col in row:
                    if col == 0:
                        temp_row.append(0)
                    else:
                        temp_row.append(temp_row0[indx_row0])
                        indx_row0 += 1
                matrix_filled.append(temp_row)
            out = np.eye(n_dim)
            for i in range(int(round(dt*12)) if int(round(dt*12)) > 0 else 1):
                out = np.matmul(out, np.array(matrix_filled))
            # out = np.linalg.matrix_power(np.array(matrix_filled), int(round(dt*12)) if int(round(dt*12)) > 0 else 1) # Assume the screening interval is at least one month
        else:
            n_dim = MPmatrix.shape[0]
            matrix_filled = np.zeros_like(matrix, dtype=np.float32)
            matrix_filled[np.where(matrix == 1)] = W
            np.fill_diagonal(matrix, 1)
            matrix = np.matmul(np.diag(1 + np.arange(n_dim)), matrix)
            for indx_row in range(n_dim):
                matrix_filled[np.where(matrix == 1+indx_row)] = Softmax(matrix_filled[np.where(matrix == 1+indx_row)])
            out = np.linalg.matrix_power(matrix_filled, int(round(dt*12)) if int(round(dt*12)) > 0 else 1) # Assume the screening interval is at least one month

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
            temp_W = W[:, temp_component]
            matrix = np.dot(matrix, ProbTransition_interval(MPmatrix, dt, temp_W))
            temp = inv[temp_component]
        else:
            dt = end - temp
            temp_W = W[:, temp_component]
            matrix = np.dot(matrix, ProbTransition_interval(MPmatrix, dt, temp_W))
            temp = inv[temp_component]


    out = matrix
    #Normalize the probability matrix
    out = np.where(out < 0, 0., out)
    out = np.where(out > 1, 1., out)
    norm = np.repeat(np.sum(out,1),out.shape[0]).reshape(out.shape)
    out = out/norm
    return out

# Joint loglikelihood of both observations and corresponding states
def Loglikelihood(indx, Z, States, Pars, verbose = False):
    patient_states = States[Z][indx]
    Alpha = Pars[0][Z]
    Eta = Pars[1][Z]
    W = Pars[2][Z]
    C = Pars[3][Z]
    MPmatrix = MPmatrixs[Z]
    res = Loglikelihood_group(indx, patient_states, Alpha, Eta, W, C, MPmatrix, verbose)
    return res

# loglikelihood of states (no observations)
def Loglikelihood_states(indx, Z, States, Pars, verbose = False):
    patient_states = States[Z][indx]
    Alpha = Pars[0][Z]
    Eta = Pars[1][Z]
    W = Pars[2][Z]
    C = Pars[3][Z]
    MPmatrix = MPmatrixs[Z]
    res = Loglikelihood_group_states(indx, patient_states, Alpha, Eta, W, C, MPmatrix, verbose)
    return res

def Loglikelihood_obs(indx, Z, States, Pars, verbose = False):
    patient_states = States[Z][indx]
    Alpha = Pars[0][Z]
    Eta = Pars[1][Z]
    W = Pars[2][Z]
    C = Pars[3][Z]
    MPmatrix = MPmatrixs[Z]
    res = Loglikelihood_group_obs(indx, patient_states, Alpha, Eta, W, C, MPmatrix, verbose)
    return res

def Loglikelihood_group (indx, patient_states, Alpha, Eta, W, C, MPmatrix, verbose = False):
    patient_ages = ages[indx]
    patient_tests = testTypes[indx]
    patient_observations = observations[indx]
    patient_treatment_indx = treatment_indx[indx]
    censor_age = censor_ages[indx]
    death_state = death_states[indx]

    ### Loglikelihood of the first state
    patient_state = patient_states[0]
    patient_age = patient_ages[0]
    patient_observation = patient_observations[0]
    patient_test = patient_tests[0]
    loglikelihood = np.log(ddirichlet_categorical(patient_state, np.exp(C[:, Age2Comp(patient_age, inv)])))
    ### Loglikelihood for the obseration of the first state
    loglikelihood += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[patient_state,:])))
    for k in xrange(nTests):
        if k == 2:
            loglikelihood += np.log(ddirichlet_mutilnominal(patient_observation[k,:2], np.exp(Alpha[k][patient_state,:])))
        else:
            loglikelihood += np.log(ddirichlet_mutilnominal(patient_observation[k,:], np.exp(Alpha[k][patient_state,:])))
    
    if verbose:
        print "1", loglikelihood

    ### 
    for j in xrange(1, len(patient_ages)):
        patient_state = patient_states[j]
        patient_age = patient_ages[j]
        patient_observation = patient_observations[j]
        patient_test = patient_tests[j]
        #### transition probability
        p_transition = ProbTransition(MPmatrix, W, patient_ages[j-1], patient_age, inv)
        if j-1 in patient_treatment_indx:
            loglikelihood += np.log(p_transition[0, patient_state])
        else:
            loglikelihood += np.log(p_transition[patient_states[j-1], patient_state])
            # Debug
            # if loglikelihood == -np.infty:
            #     print (patient_states[j-1], patient_state)
            #     print (MPmatrix)
            #     print (patient_age-patient_ages[j-1], W)
            #     print (ProbTransition(MPmatrix, W, patient_ages[j-1], patient_age, inv))
            #     import pdb
            #     pdb.set_trace()


        if verbose:
            print "2.1", loglikelihood

        #### observation likelihood
        loglikelihood += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[patient_state,:])))

        if verbose:
            print "2.2", loglikelihood

        for k in xrange(nTests):
            if k == 2:
                loglikelihood += np.log(ddirichlet_mutilnominal(patient_observation[k,:2], np.exp(Alpha[k][patient_state,:])))
            else:
                loglikelihood += np.log(ddirichlet_mutilnominal(patient_observation[k,:], np.exp(Alpha[k][patient_state,:])))
 
        if verbose:
            print "2.3", loglikelihood    

    if censor_age < patient_ages[-1]:
        # this can happen due to some rounding errors when death is very close to last screening.
        # Just move the censor date a few month after last visit.
        censor_age = patient_ages[-1] + 0.25
    p_transition = ProbTransition(MPmatrix, W, patient_ages[-1], censor_age, inv)
    if death_state > 0: # this means censor age is age of 'death', not end of observations.
        if (len(patient_ages) - 1) in patient_treatment_indx:
            loglikelihood += np.log(p_transition[0,-1])
        else:
            loglikelihood += np.log(p_transition[patient_states[-1],-1])
    else: # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
        if (len(patient_ages) - 1) in patient_treatment_indx:
            loglikelihood += np.log(1. - p_transition[0,-1])
        else:
            loglikelihood += np.log(1. - p_transition[patient_states[-1],-1])

    # Debug
    # if loglikelihood == -np.infty:
    #     print (patient_states, death_state)
    #     print (MPmatrix)
    #     print (censor_age, patient_ages, W[:,3])
    #     print (ProbTransition(MPmatrix, W, patient_ages[-1], censor_age, inv))
    #     print (ProbTransition_interval(MPmatrix, censor_age-patient_ages[-1], W[:,3]))
    #     import pdb
    #     pdb.set_trace()

    if verbose:
        print MPmatrix
        print W
        print patient_ages[-1]
        print censor_age
        print inv
        print "2.3", loglikelihood
            
    return loglikelihood

def Loglikelihood_group_states (indx, patient_states, Alpha, Eta, W, C, MPmatrix, verbose = False):
    patient_ages = ages[indx]
    patient_treatment_indx = treatment_indx[indx]
    censor_age = censor_ages[indx]
    death_state = death_states[indx]

    ### Loglikelihood of the first state
    patient_state = patient_states[0]
    patient_age = patient_ages[0]
    loglikelihood = np.log(ddirichlet_categorical(patient_state, np.exp(C[:, Age2Comp(patient_age, inv)])))

    for j in xrange(1, len(patient_ages)):
        patient_state = patient_states[j]
        patient_age = patient_ages[j]
        #### transition probability
        p_transition = ProbTransition(MPmatrix, W, patient_ages[j-1], patient_age, inv)
        if j-1 in patient_treatment_indx:
            loglikelihood += np.log(p_transition[0, patient_state])
        else:
            loglikelihood += np.log(p_transition[patient_states[j-1], patient_state])

        if verbose:
            print "1", loglikelihood
            
    return loglikelihood

def Loglikelihood_group_obs(indx, patient_states, Alpha, Eta, W, C, MPmatrix, verbose = False):
    patient_ages = ages[indx]
    patient_tests = testTypes[indx]
    patient_observations = observations[indx]
    patient_treatment_indx = treatment_indx[indx]
    censor_age = censor_ages[indx]
    death_state = death_states[indx]
    ### 
    loglikelihood = 0
    for j in xrange(len(patient_ages)):
        patient_state = patient_states[j]
        patient_age = patient_ages[j]
        patient_observation = patient_observations[j]
        patient_test = patient_tests[j]

        #### observation likelihood
        loglikelihood += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[patient_state,:])))

        if verbose:
            print "2.2", loglikelihood

        for k in xrange(nTests):
            if k == 2:
                loglikelihood += np.log(ddirichlet_mutilnominal(patient_observation[k,:2], np.exp(Alpha[k][patient_state,:])))
            else:
                loglikelihood += np.log(ddirichlet_mutilnominal(patient_observation[k,:], np.exp(Alpha[k][patient_state,:])))
 
        if verbose:
            print "2.3", loglikelihood    

    if censor_age < patient_ages[-1]:
        # this can happen due to some rounding errors when death is very close to last screening.
        # Just move the censor date a few month after last visit.
        censor_age = patient_ages[-1] + 0.25
    p_transition = ProbTransition(MPmatrix, W, patient_ages[-1], censor_age, inv)
    if death_state > 0: # this means censor age is age of 'death', not end of observations.
        if (len(patient_ages) - 1) in patient_treatment_indx:
            loglikelihood += np.log(p_transition[0,-1])
        else:
            loglikelihood += np.log(p_transition[patient_states[-1],-1])
    else: # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
        if (len(patient_ages) - 1) in patient_treatment_indx:
            loglikelihood += np.log(1. - p_transition[0,-1])
        else:
            loglikelihood += np.log(1. - p_transition[patient_states[-1],-1])

    return(loglikelihood)

def Best_Patient_States_Viterbi(Z, MPmatrix, patient_tests, patient_observations, patient_ages, patient_treatment_status, patient_censor_age, patient_death_state, Alpha, Eta, W, C, ind, inv, verbose = False, do_last = False):
    if Z == 0:
        nStates = 2
    else:
        nStates = 4
 
    nvisits = len(patient_ages)
    V = np.zeros([nvisits, nStates]) # V means the viterbi matrix
    B = np.zeros([nvisits, nStates]) # B means the backpointer
    patient_age = patient_ages[0]
    patient_test = patient_tests[0]
    patient_observation = patient_observations[0]
    Q = np.repeat(-np.infty, nStates) # temporary iterm
    if patient_treatment_status == 0:
        for s in xrange(nStates):
            # log(P(S0))
            Q[s] = np.log(ddirichlet_categorical(s,np.exp(C[:, Age2Comp(patient_age, inv)])))
            # log(P(O0,S0)) = log(O0|S0)+log(S0)
            Q[s] += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[s,:])))
            for k in xrange(nTests):
                if k == 2:
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observation[k,:2], np.exp(Alpha[k][s,:])))
                else:
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observation[k,:], np.exp(Alpha[k][s,:])))
        V[0, :] = Q    
    else:
        Q[0] = 0
        V[0, :] = Q

    ######################
    ### Recursion Pass ###
    ######################
    for j in xrange(1, nvisits):
        p_transition = ProbTransition(MPmatrix, W, patient_ages[j-1], patient_ages[j], inv)
        log_prob_obs = np.zeros(nStates)
        for s in xrange(nStates):
            log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], np.exp(Eta[s,:])))
            for k in xrange(nTests):
                if k == 2:
                    log_prob_obs[s] += np.log(ddirichlet_mutilnominal(patient_observations[j][k,:2], np.exp(Alpha[k][s,:])))
                else:
                    log_prob_obs[s] += np.log(ddirichlet_mutilnominal(patient_observations[j][k,:], np.exp(Alpha[k][s,:])))
        
        log_proportional_prob = np.repeat(V[j-1,:],nStates).reshape([nStates,nStates]) + np.transpose(np.repeat(log_prob_obs,nStates).reshape([nStates,nStates])) + np.log(p_transition[:nStates,:nStates])
        
        if verbose:
            print "2.1", p_transition
            print "2.2", log_prob_obs
            print "2.3", log_proportional_prob
            

        if j == nvisits - 1 and do_last:
            # Add the censor statue
            if patient_censor_age < patient_ages[-1]:
            # this can happen due to some rounding errors when death is very close to last screening.
            # Just move the censor date a few month after last visit.
                patient_censor_age = patient_ages[-1] + 0.25
            p_transition = ProbTransition(MPmatrix, W, patient_ages[-1], patient_censor_age, inv)
            if patient_death_state > 0: # this means censor age is age of 'death', not end of observations.
                log_proportional_prob += np.repeat(np.log(p_transition[:nStates,-1]), nStates).reshape([nStates,nStates]).T
            else: # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
                log_proportional_prob += np.repeat(np.log(1. - p_transition[:nStates,-1]), nStates).reshape([nStates, nStates]).T

            log_proportional_prob = np.where(log_proportional_prob != log_proportional_prob, -np.infty, log_proportional_prob)

            if verbose:
                print "2.4", log_proportional_prob


        for s in xrange(nStates):
            V[j, s] = np.max(log_proportional_prob[:, s])
            B[j, s] = np.argmax(log_proportional_prob[:, s])

    ####################
    ### Special case ###
    ####################
    if nvisits == 1 and do_last:
        # Add the censor statue
        if patient_censor_age < patient_ages[-1]:
        # this can happen due to some rounding errors when death is very close to last screening.
        # Just move the censor date a few month after last visit.
            patient_censor_age = patient_ages[-1] + 0.25
        p_transition = ProbTransition(MPmatrix, W, patient_ages[-1], patient_censor_age, inv)
        if patient_death_state > 0: # this means censor age is age of 'death', not end of observations.
            V[0, :] += np.log(p_transition[:nStates,-1])
        else: # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
            V[0, :] += np.log(1. - p_transition[:nStates,-1])

        if verbose:
            print "2.4", V[0, :]

        return np.array([np.argmax(V[0, :])]), np.max(V[0, :])      


    #################
    ### Backtrace ###
    #################
    best_states = np.zeros(nvisits).astype(int)
    if nvisits == 1:
        best_states[-1] = np.argmax(V[0, :])
    else:
        best_states[-1] = np.argmax(V[-1, :])
        for j in xrange(nvisits-1):    
            best_states[-j-2] = (B[-j-1, best_states[-j-1]])

    if verbose:
        print ("V: {}, B: {}, best_states: {}".format(V, B, best_states))

    return best_states, np.max(V[-1, :])

def Best_Patient_States_Viterbi0(Z, MPmatrix, patient_ages, patient_treatment_status, Alpha, Eta, W, C, ind, inv, verbose = False):
    if Z == 0:
        nStates = 2
    else:
        nStates = 4
 
    nvisits = len(patient_ages)
    # V(t, s) = max_(s_0,...,s_t-1)P(s_0,...,s_t-1,o_0,...o_t,s_t = s|-)
    V = np.zeros([nvisits, nStates]) # V means the viterbi matrix
    B = np.zeros([nvisits, nStates]) # B means the backpointer
    patient_age = patient_ages[0]
    Q = np.repeat(-np.infty, nStates) # temporary iterm
    if patient_treatment_status == 0:
        for s in xrange(nStates):
            # log(P(S0))
            Q[s] = np.log(ddirichlet_categorical(s,np.exp(C[:, Age2Comp(patient_age, inv)])))
            # # log(P(O0,S0)) = log(O0|S0)+log(S0)
            # Q[s] += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[s,:])))
            # for k in xrange(nTests):
            #     if k == 2:
            #         Q[s] += np.log(ddirichlet_mutilnominal(patient_observation[k,:2], np.exp(Alpha[k][s,:])))
            #     else:
            #         Q[s] += np.log(ddirichlet_mutilnominal(patient_observation[k,:], np.exp(Alpha[k][s,:])))
        V[0, :] = Q    
    else:
        Q[0] = 0
        V[0, :] = Q

    ######################
    ### Recursion Pass ###
    ######################
    for j in xrange(1, nvisits):
        p_transition = ProbTransition(MPmatrix, W, patient_ages[j-1], patient_ages[j], inv)
        # log_prob_obs = np.zeros(nStates)
        # for s in xrange(nStates):
        #     log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], np.exp(Eta[s,:])))
        #     for k in xrange(nTests):
        #         if k == 2:
        #             log_prob_obs[s] += np.log(ddirichlet_mutilnominal(patient_observations[j][k,:2], np.exp(Alpha[k][s,:])))
        #         else:
        #             log_prob_obs[s] += np.log(ddirichlet_mutilnominal(patient_observations[j][k,:], np.exp(Alpha[k][s,:])))
        # 
        # log_proportional_prob = np.repeat(V[j-1,:],nStates).reshape([nStates,nStates]) + np.transpose(np.repeat(log_prob_obs,nStates).reshape([nStates,nStates])) + np.log(p_transition[:nStates,:nStates])
        log_proportional_prob = np.repeat(V[j-1,:],nStates).reshape([nStates,nStates]) + np.log(p_transition[:nStates,:nStates])

        # if verbose:
        #     print "2.1", p_transition
        #     print "2.2", log_prob_obs
        #     print "2.3", log_proportional_prob
            

        # if j == nvisits - 1 and do_last:
        #     # Add the censor statue
        #     if patient_censor_age < patient_ages[-1]:
        #     # this can happen due to some rounding errors when death is very close to last screening.
        #     # Just move the censor date a few month after last visit.
        #         patient_censor_age = patient_ages[-1] + 0.25
        #     p_transition = ProbTransition(MPmatrix, W, patient_ages[-1], patient_censor_age, inv)
        #     if patient_death_state > 0: # this means censor age is age of 'death', not end of observations.
        #         log_proportional_prob += np.repeat(np.log(p_transition[:nStates,-1]), nStates).reshape([nStates,nStates]).T
        #     else: # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
        #         log_proportional_prob += np.repeat(np.log(1. - p_transition[:nStates,-1]), nStates).reshape([nStates, nStates]).T

        #     log_proportional_prob = np.where(log_proportional_prob != log_proportional_prob, -np.infty, log_proportional_prob)

        #     if verbose:
        #         print "2.4", log_proportional_prob


        for s in xrange(nStates):
            V[j, s] = np.max(log_proportional_prob[:, s])
            B[j, s] = np.argmax(log_proportional_prob[:, s])

    # ####################
    # ### Special case ###
    # ####################
    # if nvisits == 1 and do_last:
    #     # Add the censor statue
    #     if patient_censor_age < patient_ages[-1]:
    #     # this can happen due to some rounding errors when death is very close to last screening.
    #     # Just move the censor date a few month after last visit.
    #         patient_censor_age = patient_ages[-1] + 0.25
    #     p_transition = ProbTransition(MPmatrix, W, patient_ages[-1], patient_censor_age, inv)
    #     if patient_death_state > 0: # this means censor age is age of 'death', not end of observations.
    #         V[0, :] += np.log(p_transition[:nStates,-1])
    #     else: # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
    #         V[0, :] += np.log(1. - p_transition[:nStates,-1])

    #     if verbose:
    #         print "2.4", V[0, :]

    #     return np.array([np.argmax(V[0, :])]), np.max(V[0, :])      


    #################
    ### Backtrace ###
    #################
    best_states = np.zeros(nvisits).astype(int)
    if nvisits == 1:
        best_states[-1] = np.argmax(V[0, :])
    else:
        best_states[-1] = np.argmax(V[-1, :])
        for j in xrange(nvisits-1):    
            best_states[-j-2] = (B[-j-1, best_states[-j-1]])

    if verbose:
        print ("V: {}, B: {}, best_states: {}".format(V, B, best_states))

    return best_states, np.max(V[-1, :])

def Best_Patient_States(Z, MPmatrix, patient_tests, patient_observations, patient_ages, patient_treatment_status, Alpha, Eta, W, C, ind, inv, verbose = False):   ####verbose is used for debug.
    if Z == 0:
        nStates = 2
    else:
        nStates = 4

    ##########################
    ### Initial conditions ###
    ##########################
    Q = np.zeros(nStates)
    patient_age = patient_ages[0]
    patient_test = patient_tests[0]
    patient_observation = patient_observations[0]
    if patient_treatment_status == 0:
        for s in xrange(nStates):
            # log(P(S0))
            Q[s] = np.log(ddirichlet_categorical(s,np.exp(C[:, Age2Comp(patient_age, inv)])))
            # log(P(O0,S0)) = log(O0|S0)+log(S0)
            Q[s] += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[s,:])))
            for k in xrange(nTests):
                if k == 2:
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observations[j][k,:2], np.exp(Alpha[k][s,:])))
                else:
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observations[j][k,:], np.exp(Alpha[k][s,:])))
            # P(S0|O0)
        Q = np.exp(Q)/np.sum(np.exp(Q))
    else:
        Q[0] = 1
    log_Q = np.log(Q)

    ####################
    ### Forward Pass ###
    ####################
    # P_forward_matrices P(Sj-1, Sj|Oj)
    P_forward_matrices = [ np.zeros([nStates, nStates]) for patient_age in patient_ages]
    for j in xrange(1,len(patient_ages)):
        p_transition = ProbTransition(MPmatrix, W, patient_ages[j-1], patient_ages[j], inv)
        log_prob_obs = np.zeros(nStates)
        for s in xrange(nStates):
            log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], np.exp(Eta[s,:])))
            for k in xrange(nTests):
                if k == 2:
                    log_prob_obs[s] += np.log(ddirichlet_mutilnominal(patient_observation[k,:2], np.exp(Alpha[k][s,:])))
                else:
                    log_prob_obs[s] += np.log(ddirichlet_mutilnominal(patient_observation[k,:], np.exp(Alpha[k][s,:])))
        log_proportional_prob = np.repeat(log_Q,nStates).reshape([nStates,nStates]) + np.transpose(np.repeat(log_prob_obs,nStates).reshape([nStates,nStates])) + np.log(p_transition[:nStates,:nStates])
        proportional_prob = np.exp(log_proportional_prob)
        #
        P_forward_matrices[j] = np.exp( log_proportional_prob - np.log(np.sum(proportional_prob) ) )
        #
        Q = np.sum(P_forward_matrices[j],0)
        log_Q = np.log(Q)

    #####################
    ### Backward Pass ###
    #####################
    nvisits = len(patient_ages)
    best_states = np.zeros(nvisits).astype(int)
    if nvisits == 1:
        best_states[0] = np.argmax(Q)
    else:
        maxindex = np.unravel_index(P_forward_matrices[-1].argmax(),P_forward_matrices[-1].shape)
        best_states[-1] = maxindex[1]
        best_states[-2] = maxindex[0]
    for j in xrange(2,nvisits):
        best_states[-j-1] = np.argmax(P_forward_matrices[-j][:, best_states[-j]]) #Rui's
        # best_states[-j-1] = np.argmax(q_backward) # original from Scott 2002
    return best_states

# Find the best patient states given observations 
def Best_Patient_States_treatment(Z, MPmatrixs, patient_tests, patient_observations, patient_ages, patient_treatment_indx, patient_censor_age, patient_death_state, currPars, ind, inv, verbose = False):
    MPmatrix = MPmatrixs[Z]
    Alpha = currPars[0][Z]
    Eta = currPars[1][Z]
    W = currPars[2][Z]
    C = currPars[3][Z]

    if len(patient_treatment_indx) == 1:
        j = patient_treatment_indx[0]
        bs0, bl0 = Best_Patient_States_Viterbi(Z, MPmatrix, patient_tests[:(j+1)], patient_observations[:(j+1)], patient_ages[:(j+1)], 0, patient_censor_age, patient_death_state, Alpha, Eta, W, C, ind, inv, verbose)
        bs1, bl1 = Best_Patient_States_Viterbi(Z, MPmatrix, patient_tests[j:], patient_observations[j:], patient_ages[j:], 1, patient_censor_age, patient_death_state, Alpha, Eta, W, C, ind, inv, verbose, do_last = True)
        best_states = np.concatenate([bs0,bs1[1:]])
        best_loglikelihood = bl0 + bl1

    elif len(patient_treatment_indx) > 1:
        best_states_list = []
        j = patient_treatment_indx[0]
        bs0, bl0 = Best_Patient_States_Viterbi(Z, MPmatrix, patient_tests[:(j+1)], patient_observations[:(j+1)], patient_ages[:(j+1)], 0, patient_censor_age, patient_death_state, Alpha, Eta, W, C, ind, inv, verbose)
        best_states_list.append(bs0)
        best_loglikelihood = bl0
        for i in xrange(len(patient_treatment_indx)-1):
            j = patient_treatment_indx[i]
            k = patient_treatment_indx[i+1] + 1
            bsi, bli = Best_Patient_States_Viterbi(Z, MPmatrix, patient_tests[j:k], patient_observations[j:k], patient_ages[j:k], 1, patient_censor_age, patient_death_state, Alpha, Eta, W, C, ind, inv, verbose)
            best_states_list.append(bsi[1:])
            best_loglikelihood += bli
        j = patient_treatment_indx[-1]
        bs1, bl1 = Best_Patient_States_Viterbi(Z, MPmatrix, patient_tests[j:], patient_observations[j:], patient_ages[j:], 1, patient_censor_age, patient_death_state, Alpha, Eta, W, C, ind, inv, verbose, do_last = True)
        best_states_list.append(bs1[1:])
        best_loglikelihood += bl1
        best_states = np.concatenate(best_states_list)
    else:
        best_states, best_loglikelihood = Best_Patient_States_Viterbi(Z, MPmatrix, patient_tests, patient_observations, patient_ages, 0, patient_censor_age, patient_death_state, Alpha, Eta, W, C, ind, inv, verbose, do_last = True)
    #########
    # check #
    #########
    if len(best_states) != len(patient_ages):
        print 'Error in Best_Patient_States_treatment: best state vector length does not match patient_ages vector length.'

    return best_states, best_loglikelihood

# Find the best patient states given no observations Working !!!!!!!!!!!!!!!!!!!!!
def Best_Patient_States_treatment0(Z, MPmatrixs, patient_ages, patient_treatment_indx, currPars, ind, inv, verbose = False):
    MPmatrix = MPmatrixs[Z]
    Alpha = currPars[0][Z]
    Eta = currPars[1][Z]
    W = currPars[2][Z]
    C = currPars[3][Z]
   
    if len(patient_treatment_indx) == 1:
        j = patient_treatment_indx[0]
        bs0, bl0 = Best_Patient_States_Viterbi0(Z, MPmatrix, patient_ages[:(j+1)], 0, Alpha, Eta, W, C, ind, inv, verbose)
        bs1, bl1 = Best_Patient_States_Viterbi0(Z, MPmatrix, patient_ages[j:], 1, Alpha, Eta, W, C, ind, inv, verbose)
        best_states = np.concatenate([bs0,bs1[1:]])
        best_loglikelihood = bl0 + bl1

    elif len(patient_treatment_indx) > 1:
        best_states_list = []
        j = patient_treatment_indx[0]
        bs0, bl0 = Best_Patient_States_Viterbi0(Z, MPmatrix, patient_ages[:(j+1)], 0, Alpha, Eta, W, C, ind, inv, verbose)
        best_states_list.append(bs0)
        best_loglikelihood = bl0
        for i in xrange(len(patient_treatment_indx)-1):
            j = patient_treatment_indx[i]
            k = patient_treatment_indx[i+1] + 1
            bsi, bli = Best_Patient_States_Viterbi0(Z, MPmatrix, patient_ages[j:k], 1, Alpha, Eta, W, C, ind, inv, verbose)
            best_states_list.append(bsi[1:])
            best_loglikelihood += bli
        j = patient_treatment_indx[-1]
        bs1, bl1 = Best_Patient_States_Viterbi0(Z, MPmatrix, patient_ages[j:], 1, Alpha, Eta, W, C, ind, inv, verbose)
        best_states_list.append(bs1[1:])
        best_loglikelihood += bl1
        best_states = np.concatenate(best_states_list)
    else:
        best_states, best_loglikelihood = Best_Patient_States_Viterbi0(Z, MPmatrix, patient_ages, 0, Alpha, Eta, W, C, ind, inv, verbose)
    #########
    # check #
    #########
    if len(best_states) != len(patient_ages):
        print 'Error in Best_Patient_States_treatment: best state vector length does not match patient_ages vector length.'

    return best_states, best_loglikelihood

# Joint negative loglikelihood of both observations and all states
def NegativeLogLikelihood(parameter_vector):
    Z_pos = currZ_pos
    States = currStates
    # States0 = currStates0
    Alpha, Eta, W, C = ParVec2ParList(parameter_vector, n_inv, ind)
    Pars = [Alpha, Eta, W, C]
    # initial conditions #
    loglikelihood = 0
    states_0 = States[0]
    states_1 = States[1]

    for patient_indx in xrange(len(ages)):
        loglik_0 = Loglikelihood(patient_indx, 0, States, Pars)
        # loglik_state1 = Loglikelihood_states(patient_indx, 1, States0, Pars)
        loglik_1 = Loglikelihood(patient_indx, 1, States, Pars)
        # loglik_state0 = Loglikelihood_states(patient_indx, 0, States0, Pars)
        
        # loglikelihood += (1- Z_pos[patient_indx])*(loglik_0 + loglik_state1) + Z_pos[patient_indx]*(loglik_1 + loglik_state0)
        loglikelihood += (1- Z_pos[patient_indx])*(loglik_0 + np.log(1-p)) + Z_pos[patient_indx]*(loglik_1 + np.log(p))
    return -loglikelihood

def NegativeLogLikelihood_shared(parameter_vector_shared):
    parameter_vector = ParVecShared2ParVec(parameter_vector_shared, n_inv, ind)
    return NegativeLogLikelihood(parameter_vector)

# Joint negative loglikelihood of both observations and all states for parallel version
def NegativeLogLikelihood_caller(parameter_vector, verbose = False): 
    ''' This simply evaluates the log likelihood of the model given ALL data, even latent variables.
        Some important values and data structures:
        S = number of underlying states. Usually 4.
        K = number of diagnostic tests.  Usually 3.
        N = number of patients.
        O[i] = number of observations of patient i

        Note that we are assuming each test outputs S possible results corresponding to trying to detect the underlying state.
        In some cases this is not true so we have to implement structural zeros in the emission probabilities for these cases.

        'model_parameters' is a vector of all model parameters.


        'times', 'testTypes', 'observations', 'states', 'ages', 'treatment_indx' are all lists of length N
        'times[i]' is an array of length O[i]
        'testTypes[i]' is a list of length O[i], each element of which is an array of length K
        'observations[i]' is a list of length O[i], each element of which is an array of size KxS
        'frailty' is a list or array of length N with 0 or 1 elements.
        'states' is a list of length N, each element of which is an array of length O[i].
    '''
    stop[0] = comm.bcast(stop[0], 0)
    summ = 0
    if stop[0] == 0:
        parameter_vector = comm.bcast(parameter_vector, 0)
        negativeloglikelihood = NegativeLogLikelihood(parameter_vector)
        if verbose:
            print ("NegativeLogLikelihood for {}th process = {}".format(rank, negativeloglikelihood))
        summ = comm.reduce(negativeloglikelihood, op = MPI.SUM, root = 0)
    if rank == 0:
        return summ
    else:
        return 0

def NegativeLogLikelihood_shared_caller(parameter_vector_shared):
    # Convert parameter_vector_shared to parameter_vector
    parameter_vector = ParVecShared2ParVec(parameter_vector_shared, n_inv, ind)
    res = NegativeLogLikelihood_caller(parameter_vector)
    return res
    
def NegativeLoglikelihood_shared_grad_caller(parameter_vector_shared, verbose = False):
    stop[0] = comm.bcast(stop[0], 0)
    negativeloglikelihood_grad_list = []
    if stop[0] == 0:
        parameter_vector_shared = comm.bcast(parameter_vector_shared, 0)
        negativeloglikelihood_grad_f = grad(NegativeLogLikelihood_shared)
        negativeloglikelihood_grad = negativeloglikelihood_grad_f(parameter_vector_shared)
        if verbose:
            print ("NegativeLogLikelihood gradient for {}th process = {}".format(rank, negativeloglikelihood_grad))
        negativeloglikelihood_grad_list = comm.gather(negativeloglikelihood_grad, root = 0)

    if rank == 0:
        summ = sum(negativeloglikelihood_grad_list)
        if verbose:
            print ("negativeloglikelihood gradient is {}".format(summ))
        return summ
    else:
        return 0 

def Compute_pos_Z_state(p, verbose = False): ### given state Z | S -
    global currZ_pos, currZ_pos_list
    # It is a Bernouli(p)
    t_z = time.clock()
    if counter == 0:
        # Rough estimation
        currZ_pos = [.5 for nvs in xrange(nPatients)]
    else:
        prevZ_pos = currZ_pos
        currZ_pos = []
        for indx in xrange(nPatients):
            loglik_0 = Loglikelihood_obs(indx, 0, currStates, currPars)
            loglik_1 = Loglikelihood_obs(indx, 1, currStates, currPars)
            tilde_p = np.exp(np.log(p)+loglik_1 - np.log((1-p)*np.exp(loglik_0) + p*np.exp(loglik_1)))
            currZ_pos.append(tilde_p)

    comm.Barrier()
    currZ_pos_list = comm.gather(currZ_pos, 0)
    if rank == 0:
        print('Compute the posterior of Z costs {}'.format(time.clock() - t_z))
    ## print the Negative log likelihood
    if counter > 0:
        stop = [0]
        if rank == 0:
            print ('Current negative loglikelihood is: {}'.format(NegativeLogLikelihood_caller(curr_parameter_vector, verbose)))
            # print ('Current probabilities of Z is: {}'.format(currZ_pos_list))
        else:
            NegativeLogLikelihood_caller(curr_parameter_vector)
    return 0

def Compute_pos_Z(p, verbose = False): ### given no state Z | -
    global currZ_pos, currZ_pos_list
    # It is a Bernouli(p)
    t_z = time.clock()
    if counter == 0:
        # Rough estimation
        currZ_pos = [.5 for nvs in xrange(nPatients)]
    else:
        prevZ_pos = currZ_pos
        currZ_pos = []
        for indx in xrange(nPatients):
            loglik_0 = Loglikelihood_obs0(indx, 0, currPars)
            loglik_1 = Loglikelihood_obs0(indx, 1, currPars)
            tilde_p = np.exp(np.log(p)+loglik_1 - np.log((1-p)*np.exp(loglik_0) + p*np.exp(loglik_1)))
            currZ_pos.append(tilde_p)

    comm.Barrier()
    currZ_pos_list = comm.gather(currZ_pos, 0)
    if rank == 0:
        print('Compute the posterior of Z costs {}'.format(time.clock() - t_z))
    ## print the Negative log likelihood
    if counter > 0:
        stop = [0]
        if rank == 0:
            print ('Current negative loglikelihood is: {}'.format(NegativeLogLikelihood_caller(curr_parameter_vector, verbose)))
            # print ('Current probabilities of Z is: {}'.format(currZ_pos_list))
        else:
            NegativeLogLikelihood_caller(curr_parameter_vector)
    return 0

def Compute_Z(verbose = False):
    global currZ_pos, currZ_pos_list
    t_z = time.clock()
    if counter == 0:
        currZ_pos = [.5 for nvs in xrange(nPatients)]
    else:
        prevZ_pos = currZ_pos
        currZ_pos = []
        for indx in xrange(nPatients):
            loglik_0 = Loglikelihood(indx, 0, currStates, currPars)
            loglik_1 = Loglikelihood(indx, 1, currStates, currPars)
            if loglik_1 > loglik_0:
                p = 1
            else:
                p = 0
            currZ_pos.append(p)

            # #### test
            # currloglik = np.log((1- p)*np.exp(loglik_0) + p*np.exp(loglik_1))
            # prevloglik = np.log((1-prevZ_pos[indx])*np.exp(loglik_0) + prevZ_pos[indx]*np.exp(loglik_1))
            # if currloglik < prevloglik:
            #     print ("For rank {}, the {}th patient has currloglik {} and prevloglik {}".format(rank, indx, currloglik, prevloglik))
            #     print loglik_0
            #     print loglik_1
            #     print ("p = {} and prev_p = {}".format(p, prevZ_pos[indx]))
            # ####

    comm.Barrier()
    currZ_pos_list = comm.gather(currZ_pos, 0)
    if rank == 0:
        print('Compute the posterior of Z costs {}'.format(time.clock() - t_z))
    ## print the Negative log likelihood
    if counter > 0:
        stop = [0]
        if rank == 0:
            print ('Current negative loglikelihood is: {}'.format(NegativeLogLikelihood_caller(curr_parameter_vector, currZ_pos, currStates, stop, verbose)))
            # print ('Current probabilities of Z is: {}'.format(currZ_pos_list))
        else:
            NegativeLogLikelihood_caller(curr_parameter_vector, currZ_pos, currStates, stop)
    return 0

def Update_states_single(verbose = False):  ### S_z | z 
    global currStates, currStates_list, stop
    if rank == 0:
        t_s = time.clock()

    currStates_0 = []
    currLoglikelihood_0 = []
    currStates_1 = []   
    currLoglikelihood_1 = []
    if counter == 0:     
        # Rough estimation
        for patient_observations in observations:
            patient_states = np.array([np.argmax(np.sum(patient_observation, axis = 0))  for patient_observation in patient_observations])
            #### Correction for initial states
            for i in range(len(patient_states)):
                if patient_states[i] == 3:
                    patient_states[i] = 2

            currStates_0.append((patient_states > 0).astype(int))
            if model == "continuous":
                currStates_1.append(patient_states)
            elif model == "discrete":
                currStates_1.append((patient_states > 0).astype(int))
    else:
        # #### test
        prevStates_0 = currStates[0]
        prevStates_1 = currStates[1]
        prevStates = [prevStates_0, prevStates_1]
        patient_indx = -1
        # ####
        for patient_tests, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state in zip(testTypes, observations, ages, treatment_indx, censor_ages, death_states):
            #### test
            patient_indx += 1
            ####
            currStates = [currStates_0, currStates_1]
            patient_states, patient_loglikelihood = Best_Patient_States_treatment(0, MPmatrixs, patient_tests, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state, currPars, ind, inv, verbose = False)
            currStates_0.append(patient_states)
            currLoglikelihood_0.append(patient_loglikelihood)
            patient_states, patient_loglikelihood = Best_Patient_States_treatment(1, MPmatrixs, patient_tests, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state, currPars, ind, inv, verbose = False)
            currStates_1.append(patient_states)
            currLoglikelihood_1.append(patient_loglikelihood)

            #### test
            currStates = [currStates_0, currStates_1]
            # print (currStates, patient_treatment_indx)
            # l_prev = Loglikelihood(patient_indx, 0, prevStates, currPars)
            # l_curr = Loglikelihood(patient_indx, 0, currStates, currPars)
            # print ("In rank {}. previous {}th patient's current states: {}".format(rank, patient_indx, currStates))
            # if  np.round(l_prev, 5) > np.round(l_curr, 5): 
            #     print ("In rank {}. previous {}th patient's loglikelihood in 0 category: {}".format(rank, patient_indx, l_prev))
            #     print ("In rank {}. current {}th patient's loglikelihood in 0 category: {} and {}".format(rank, patient_indx, l_curr, currLoglikelihood_0[-1]))
            # if np.round(l_curr, 5) != np.round(currLoglikelihood_0[-1], 5):
            #     print ("In rank {}. current {}th patient's treatment: {}, censor_age: {}, death_state: {}".format(rank, patient_indx, patient_treatment_indx, censor_age, death_state))
            #     print ("In rank {}. current {}th patient's loglikelihood in 0 category: {} and {}".format(rank, patient_indx, l_curr, currLoglikelihood_0[-1]))
            #     Best_Patient_States_treatment(0, MPmatrixs, patient_tests, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state, currPars, ind, inv, verbose = True)
            #     print ("########################################")
            #     Loglikelihood(patient_indx, 0, currStates, currPars, verbose = True)
            # l_prev = Loglikelihood(patient_indx, 1, prevStates, currPars)
            # l_curr = Loglikelihood(patient_indx, 1, currStates, currPars)
            # if  np.round(l_prev, 5) > np.round(l_curr, 5):     
            #     print ("In rank {}. previous {}th patient's loglikelihood in 1 category: {}".format(rank, patient_indx, l_prev))
            #     print ("In rank {}. current {}th patient's loglikelihood in 1 category: {} and {}".format(rank, patient_indx, l_curr, currLoglikelihood_1[-1]))
            # if  np.round(l_curr,5) != np.round(currLoglikelihood_1[-1], 5):
            #     print ("In rank {}. current {}th patient's treatment: {}, censor_age: {}, death_state: {}".format(rank, patient_indx, patient_treatment_indx, censor_age, death_state))
            #     print ("In rank {}. current {}th patient's loglikelihood in 1 category: {} and {}".format(rank, patient_indx, l_curr, currLoglikelihood_1[-1]))
            #     Best_Patient_States_treatment(1, MPmatrixs, patient_tests, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state, currPars, ind, inv, verbose = True)
            #     print ("########################################")
            #     Loglikelihood(patient_indx, 1, currStates, currPars, verbose = True)
            ####


    currStates = [currStates_0, currStates_1]
    currStates_list = comm.gather(currStates)
    comm.Barrier()
    if rank == 0:
        print ('Update states costs: {}'.format(time.clock() - t_s))
     ## print the Negative log likelihood
    stop = [0]
    if rank == 0:
        print ('Current negative loglikelihood is: {}'.format(NegativeLogLikelihood_caller(curr_parameter_vector, verbose)))
        # print ('Current States: \n Model 0: {} \n Model 1: {}'.format(currStates_0, currStates_1))
    else:
        NegativeLogLikelihood_caller(curr_parameter_vector, verbose)
    return 0  

def Update_states(verbose = False):   ### S | z
    global currStates, currStates_list, stop #, currStates0
    if rank == 0:
        t_s = time.clock()

    currStates_0 = [] 
    currStates_1 = []  
    # currStates0_0 = []
    # currStates0_1 = []
    if counter == 0:     
        # Rough estimation
        for patient_observations in observations:
            patient_states = np.array([np.argmax(np.sum(patient_observation, axis = 0))  for patient_observation in patient_observations])
            #### Correction for initial states
            for i in range(len(patient_states)):
                if patient_states[i] == 3:
                    patient_states[i] = 2

            currStates_0.append((patient_states > 0).astype(int))
            if model == "continuous":
                currStates_1.append(patient_states)
            elif model == "discrete":
                currStates_1.append((patient_states > 0).astype(int))
        # currStates0_0 = currStates_0[:]
        # currStates0_1 = currStates_1[:]
    else:
        # #### test
        prevStates_0 = currStates[0]
        prevStates_1 = currStates[1]
        prevStates = [prevStates_0, prevStates_1]
        patient_indx = -1
        # ####
        for patient_tests, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state in zip(testTypes, observations, ages, treatment_indx, censor_ages, death_states):
            #### test
            patient_indx += 1
            ####
            patient_states, patient_loglikelihood = Best_Patient_States_treatment(0, MPmatrixs, patient_tests, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state, currPars, ind, inv, verbose = False)
            currStates_0.append(patient_states)
            patient_states, patient_loglikelihood = Best_Patient_States_treatment(1, MPmatrixs, patient_tests, patient_observations, patient_ages, patient_treatment_indx, censor_age, death_state, currPars, ind, inv, verbose = False)
            currStates_1.append(patient_states)
            # patient_states, patient_loglikelihood = Best_Patient_States_treatment0(0, MPmatrixs, patient_ages, patient_treatment_indx, currPars, ind, inv, verbose = False)
            # currStates0_0.append(patient_states)
            # patient_states, patient_loglikelihood = Best_Patient_States_treatment0(1, MPmatrixs, patient_ages, patient_treatment_indx, currPars, ind, inv, verbose = False)
            # currStates0_1.append(patient_states)


    currStates = [currStates_0, currStates_1]
    # currStates0 = [currStates0_0, currStates0_1]
    currStates_list = comm.gather(currStates)
    comm.Barrier()
    if rank == 0:
        print ('Update states costs: {}'.format(time.clock() - t_s))
     ## print the Negative log likelihood
    stop = [0]
    if rank == 0:
        print ('Current negative loglikelihood is: {}'.format(NegativeLogLikelihood_caller(curr_parameter_vector, verbose)))
        # print ('Current States: \n Model 0: {} \n Model 1: {}'.format(currStates_0, currStates_1))
    else:
        NegativeLogLikelihood_caller(curr_parameter_vector, verbose)
    return 0 ### already know the true model indicator

def Update_all_pars(verbose = False):
    global currPars, currNegLogLik, curr_parameter_vector
    stop = [0]
    comm.Barrier()
    # break
    parameter_vector = ParList2ParVec(currPars[0], currPars[1], currPars[2], currPars[3], ind)

    if rank == 0:
        t_m = time.clock()
        res = optimize.minimize(NegativeLogLikelihood_caller,
                                x0=parameter_vector,
                                args=(currZ_pos, currStates, stop),
                                jac=None,
                                 method='L-BFGS-B',
                                options={'disp': True, 'maxiter': max_steps_optim})
        stop = [1] # Declare the NegativeLogLikelihood_caller computation stoped for optimization
        NegativeLogLikelihood_caller(parameter_vector, currZ_pos, currStates, stop)
    else:
        while stop[0] == 0:
             NegativeLogLikelihood_caller(parameter_vector, currZ_pos, currStates, stop)

    if rank == 0:
        curr_parameter_vector = res.x
        currNegLogLik = res.fun
        print ('Updating all parameters costs: ', time.clock() - t_m)
        currAlpha, currEta, currW, currC = ParVec2ParList(curr_parameter_vector, n_inv, ind)
        currPars = [currAlpha, currEta, currW, currC]
        print ("Under Model 0 \nCurrent Alpha: {}\n Current Eta: {}\n Current W: {}\n Current C:{}".format(currAlpha[0], currEta[0], currW[0], currC[0]))
        print ("Under Model 1 \nCurrent Alpha: {}\n Current Eta: {}\n Current W: {}\n Current C:{}".format(currAlpha[1], currEta[1], currW[1], currC[1]))
    ## print the Negative log likelihood
    stop = [0]
    if rank == 0:
        print ('Current negative loglikelihood is: {}'.format(NegativeLogLikelihood_caller(curr_parameter_vector, currZ_pos, currStates, stop, verbose)))
    else:
        NegativeLogLikelihood_caller(curr_parameter_vector, currZ_pos, currStates, stop)
    return 0

def Update_all_pars_shared(verbose = False):
    global currPars, currNegLogLik, curr_parameter_vector
    global stop
    stop = [0]
    comm.Barrier()
    # break
    parameter_vector = ParList2ParVec(currPars[0], currPars[1], currPars[2], currPars[3], ind)
    parameter_vector_shared = ParVec2ParVecShared(parameter_vector, n_inv, ind)

    if rank == 0:
        t_m = time.clock()
        if autograd_optim:
            res = optimize.minimize(NegativeLogLikelihood_shared_caller,
                                    x0=parameter_vector_shared,
                                    jac=NegativeLoglikelihood_shared_grad_caller,
                                    method='L-BFGS-B',
                                    options={'disp': True, 'maxiter': max_steps_optim})
        else:
            res = optimize.minimize(NegativeLogLikelihood_shared_caller,
                                    x0=parameter_vector_shared,
                                    jac=None,
                                    method='L-BFGS-B',
                                    options={'disp': True, 'maxiter': max_steps_optim})
        stop = [1] # Declare the NegativeLogLikelihood_caller computation stoped for optimization
        if autograd_optim:
            NegativeLogLikelihood_shared_caller(parameter_vector_shared)
            NegativeLoglikelihood_shared_grad_caller(parameter_vector_shared)
        else:
            NegativeLogLikelihood_shared_caller(parameter_vector_shared)
    else:
        while stop[0] == 0:
            if autograd_optim:
                NegativeLogLikelihood_shared_caller(parameter_vector_shared)
                NegativeLoglikelihood_shared_grad_caller(parameter_vector_shared)
            else:
                NegativeLogLikelihood_shared_caller(parameter_vector_shared)
                
    if rank == 0:
        curr_parameter_vector_shared = res.x
        currNegLogLik = res.fun
        print ('Updating all parameters costs: ', time.clock() - t_m)
        curr_parameter_vector = ParVecShared2ParVec(curr_parameter_vector_shared, n_inv, ind)
        currAlpha, currEta, currW, currC = ParVec2ParList(curr_parameter_vector, n_inv, ind)
        currPars = [currAlpha, currEta, currW, currC]
        print ("Under Model 0 \nCurrent Alpha: {}\n Current Eta: {}\n Current W: {}\n Current C:{}".format(currAlpha[0], currEta[0], currW[0], currC[0]))
        print ("Under Model 1 \nCurrent Alpha: {}\n Current Eta: {}\n Current W: {}\n Current C:{}".format(currAlpha[1], currEta[1], currW[1], currC[1]))
    ## print the Negative log likelihood
    stop = [0]
    if rank == 0:
        print ('Current negative loglikelihood is: {}'.format(NegativeLogLikelihood_caller(curr_parameter_vector, verbose)))
    else:
        NegativeLogLikelihood_caller(curr_parameter_vector)
    return 0

def Update_all_pars_by_group(verbose = False):
    global currPars, currNegLogLik, curr_parameter_vector

    # Update the parameters in the Model 0
    stop = [0]
    comm.Barrier()
    currPars_0 = [currPar[0] for currPar in currPars]
    parameter_vector = ParList2ParVec_group(currPars_0[0], currPars_0[1], currPars_0[2], currPars_0[3], ind)
    if rank == 0:
        t_m = time.clock()
        res = optimize.minimize(NegativeLogLikelihood_group_caller,
                                x0=parameter_vector,
                                args=(0, currPars, currZ_pos, currStates, stop),
                                jac = None,
                                method='L-BFGS-B',
                                options={'disp': True, 'maxiter': max_steps_optim})
        stop = [1]
        NegativeLogLikelihood_group_caller(parameter_vector, 0, currPars, currZ_pos, currStates, stop)
    else:
        while stop[0] == 0:
            NegativeLogLikelihood_group_caller(parameter_vector, 0, currPars, currZ_pos, currStates, stop)
    if rank == 0:    
        curr_parameter_vector_0 = res.x
        currNegLogLik_0 = res.fun
        print ('Updating parameters for Model 0 costs: ', time.clock() - t_m)
        currAlpha_0, currEta_0, currW_0, currC_0 = ParVec2ParList_group(0, curr_parameter_vector_0, n_inv, ind)
        print ("Under Model 0 \nCurrent Alpha: {}\n Current Eta: {}\n Current W: {}\n Current C:{}".format(currAlpha_0, currEta_0, currW_0, currC_0))
        currPars[0][0] = currAlpha_0
        currPars[1][0] = currEta_0
        currPars[2][0] = currW_0
        currPars[3][0] = currC_0


    # Update the parameters in the Model 1
    stop = [0]
    comm.Barrier()
    currPars_1 = [currPar[1] for currPar in currPars]
    parameter_vector = ParList2ParVec_group(currPars_1[0], currPars_1[1], currPars_1[2], currPars_1[3], ind)
    if rank == 0:
        t_m = time.clock()
        res = optimize.minimize(NegativeLogLikelihood_group_caller,
                                x0=parameter_vector,
                                args=(1, currPars, currZ_pos, currStates, stop),
                                jac = None,
                                method='L-BFGS-B',
                                options={'disp': True, 'maxiter': max_steps_optim})
        stop = [1]
        NegativeLogLikelihood_group_caller(parameter_vector, 1, currPars, currZ_pos, currStates, stop)
    else:
        while stop[0] == 0:
            NegativeLogLikelihood_group_caller(parameter_vector, 1, currPars, currZ_pos, currStates, stop)
    if rank == 0:    
        curr_parameter_vector_1 = res.x
        currNegLogLik_1 = res.fun
        print ('Updating parameters for Model 1 costs: ', time.clock() - t_m)
        currAlpha_1, currEta_1, currW_1, currC_1 = ParVec2ParList_group(1, curr_parameter_vector_1, n_inv, ind)
        print ("Under Model 1 \nCurrent Alpha: {}\n Current Eta: {}\n Current W: {}\n Current C:{}".format(currAlpha_1, currEta_1, currW_1, currC_1))

    if rank == 0:
        # Combine parameters for two models
        currNegLogLik = currNegLogLik_0 + currNegLogLik_1
        currAlpha = [currAlpha_0, currAlpha_1]
        currEta = [currEta_0, currEta_1]
        currW = [currW_0, currW_1]
        currC = [currC_0, currC_1]
        currPars = [currAlpha, currEta, currW, currC]
        curr_parameter_vector = ParList2ParVec(currAlpha, currEta, currW, currC, ind)

    ## print the Negative log likelihood
    stop = [0]
    if rank == 0:
        print ('Current negative loglikelihood is: {}'.format(NegativeLogLikelihood_caller(curr_parameter_vector, currZ_pos, currStates, stop, verbose)))
    else:
        NegativeLogLikelihood_caller(curr_parameter_vector, currZ_pos, currStates, stop)
    return 0

# loglikelihood of a single patient's observatoins for model Z (no states)
def Loglikelihood_obs0(indx, Z, Pars, verbose = False):
    Alpha = Pars[0][Z]
    Eta = Pars[1][Z]
    W = Pars[2][Z]
    C = Pars[3][Z]
    MPmatrix = MPmatrixs[Z]

    patient_ages = ages[indx]
    patient_tests = testTypes[indx]
    patient_observations = observations[indx]
    patient_treatment_indx = treatment_indx[indx]
    patient_censor_age = censor_ages[indx]
    patient_death_state = death_states[indx]

    if len(patient_treatment_indx) == 1:
        j = patient_treatment_indx[0]
        loglik0 = Loglikelihood_group_obs0(patient_tests[:(j+1)], patient_observations[:(j+1)], patient_ages[:(j+1)], 0, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose)
        loglik1 = Loglikelihood_group_obs0(patient_tests[j:], patient_observations[j:], patient_ages[j:], 1, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose, do_last = True)
        loglik = loglik0 + loglik1
    elif len(patient_treatment_indx) > 1:
        loglik = 0
        j = patient_treatment_indx[0]
        loglik0 = Loglikelihood_group_obs0(patient_tests[:(j+1)], patient_observations[:(j+1)], patient_ages[:(j+1)], 0, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose)
        loglik += loglik0
        for i in xrange(len(patient_treatment_indx)-1):
            j = patient_treatment_indx[i]
            k = patient_treatment_indx[i+1] + 1
            logliki= Loglikelihood_group_obs0(patient_tests[j:k], patient_observations[j:k], patient_ages[j:k], 1, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose)
            loglik += logliki
        j = patient_treatment_indx[-1]
        loglik1= Loglikelihood_group_obs0(patient_tests[j:], patient_observations[j:], patient_ages[j:], 1, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose, do_last = True)
        loglik += loglik1
    else:
        loglik = Loglikelihood_group_obs0(patient_tests, patient_observations, patient_ages, 0, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose, do_last = True)
    return loglik    

def Loglikelihood_group_obs0(patient_tests, patient_observations, patient_ages, patient_treatment_status, patient_censor_age, patient_death_state, Z, Alpha, Eta, W, C, ind, inv, verbose = False, do_last = False):
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
        for s in xrange(nStates):
            Q[s] = np.log(ddirichlet_categorical(s,np.exp(C[:, Age2Comp(patient_age, inv)])))
            # Q[s] = np.log(C[s, Age2Comp(patient_age, inv)])
            Q[s] += np.sum(stats.poisson.logpmf(patient_test, np.exp(Eta[s,:])))
            # Q[s] += np.sum(stats.poisson.logpmf(patient_test, Eta[s,:]))
            for k in xrange(nTests):
                if k == 2:
                    # Q[s] += multinomial_logpmf(patient_observations[0][k, :2], Alpha[k][s, :])
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observations[0][k,:2], np.exp(Alpha[k][s,:])))
                else:
                    # Q[s] += multinomial_logpmf(patient_observations[0][k, :], Alpha[k][s, :])
                    Q[s] += np.log(ddirichlet_mutilnominal(patient_observations[0][k,:], np.exp(Alpha[k][s,:])))
            # P(S0, O0)
        Q = np.exp(Q)
    else:
        Q[0] = 1
    log_Q = np.log(Q)

    ####################
    ### Forward Pass ###
    ####################
    # P_forward_matrices P(Sj-1, Sj, O0-j)
    P_forward_matrices = [ np.zeros([nStates, nStates]) for patient_age in patient_ages]
    for j in xrange(1, nvisits):
        p_transition = ProbTransition(MPmatrix, W, patient_ages[j-1], patient_ages[j], inv)
        log_prob_obs = np.zeros(nStates)
        for s in xrange(nStates):
            log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], np.exp(Eta[s,:])))
            # log_prob_obs[s] += np.sum(stats.poisson.logpmf(patient_tests[j], Eta[s,:]))
            for k in xrange(nTests):
                if k == 2:
                    # log_prob_obs[s] += multinomial_logpmf(patient_observations[j][k, :2], Alpha[k][s, :])
                    log_prob_obs[s] += np.log(ddirichlet_mutilnominal(patient_observations[j][k,:2], np.exp(Alpha[k][s,:])))               
                else:
                    # log_prob_obs[s] += multinomial_logpmf(patient_observations[j][k, :], Alpha[k][s, :])
                    log_prob_obs[s] += np.log(ddirichlet_mutilnominal(patient_observations[j][k,:], np.exp(Alpha[k][s,:])))
        
        log_P_forward_matrix = np.repeat(log_Q,nStates).reshape([nStates,nStates]) + np.transpose(np.repeat(log_prob_obs,nStates).reshape([nStates,nStates])) + np.log(p_transition[:nStates,:nStates])
        P_forward_matrix = np.exp(log_P_forward_matrix)
        #
        P_forward_matrices[j] = P_forward_matrix
        #
        Q = np.sum(P_forward_matrix,0)/np.sum(P_forward_matrix)
        log_Q = np.log(Q)

    ## P(S_T, O)
    if nvisits > 1: 
        PP = np.sum(P_forward_matrices[nvisits-1], 0)
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
        if patient_death_state > 0: # this means censor age is age of 'death', not end of observations.
            log_PP += np.log(p_transition[:nStates,-1])
        else: # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
            log_PP += np.log(1. - p_transition[:nStates,-1])

    # print ("log_PP", log_PP)

    return np.log(np.sum(np.exp(log_PP)))





if __name__ == "__main__":
#################################
####### Initialization ##########
#################################
    Initialization()

###################################
####### Modified EM section #######
###################################
    counter  = -1
    while (counter < max_steps_em):
        counter += 1
        if rank == 0:
            print ('step {}'.format(counter))

        ###############################################
        ### Compute the posterior distribution of Z ###
        ###############################################
        Compute_pos_Z(p)
        # Compute_Z() #hard assignment


        #################################
        ### Update the current states ###
        #################################
        Update_states()

        #############################
        ### Update all parameters ###
        #############################
        # Update_all_pars()

        #############################################################
        ### Update all parameters with shared emission parameters ###
        #############################################################
        t_s = time.time()
        # Update "Alpha, Eta" and "W, C" sequentially
        # ind = ["Alpha", "Eta"]
        # Update_all_pars_shared()
        # ind = ["W", "C"]
        # Update_all_pars_shared()
        # Update all parameters simutaneously
        ind = ["Alpha", "Eta", "W", "C"]
        Update_all_pars_shared()
        print("update all shared parameters costs {} s". format(time.time() - t_s))

        ######################################
        ### Update all parameters by group ###
        ######################################
        # Update_all_pars_by_group()




        ###################
        ### Save update ###
        ###################
        if rank == 0:
            currAlpha, currEta, currW, currC = currPars
            if p == 0.5:
                with open("res_" + "seed_" + str(args.bootstrap_seed), "wb") as em_res:
                    pickle.dump([counter, currZ_pos_list , currStates_list, currAlpha, currEta, currW, currC, currNegLogLik, inv], em_res)
            else:
                print("save results in dictory {}".format(os.getcwd()))
                with open("res_" + "p_" + str(p) + "seed_" + str(args.bootstrap_seed), "wb") as em_res:
                    pickle.dump([counter, currZ_pos_list , currStates_list, currAlpha, currEta, currW, currC, currNegLogLik, inv], em_res)

        ####################################
        ### Broadcast updated parameters ###
        ####################################
        comm.Barrier()
        # Distribute new parameters in our model
        currPars = comm.bcast(currPars, 0)