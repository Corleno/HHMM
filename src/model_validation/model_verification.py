'''

    model verification using the goodness of fit test statistics

'''
import sys
import os

#sys.path.append('/Users/ruimeng/Dropbox/Research/LLNL/Projects/GeneralStateModel')
sys.path.append('.')
sys.path.append('../')

### Libraries ###
import pdb
import numpy as np
import scipy as sci
from scipy import stats
from scipy.linalg import expm, expm2, expm3
import scipy.optimize as optimize
import scipy.stats as stats
from scipy.stats import poisson
try:
    import cPickle as pickle
except:
    import pickle
import time
import matplotlib.pyplot as plt


### User Defined Libraries ###
import mcmcFunctions as fun
import LoadDataWithLists as load
import FBalg as fb
import GetCurrentValues as getValues
# import EMalg as em
from ParameterFormatter import *


# uncomment if doing parallel version:
from  mpi4py import MPI
from scipy.optimize import approx_fprime

def ParList2ParVec_simplified(Alpha, Eta, W, Gamma, Zeta, A, B, C, ind , nDiseaseStates=4): # ind specify which parameters should be considered into optimization

    parameter_vector = np.array([])
    cons = []
    indxs_from = 0
    indxs_to = 0
    indxs_from_list = []
    indxs_to_list = []

    if "Alpha" in ind:
        a0 = Alpha[0]
        print (a0)
        a1 = Alpha[1]
        theta0 = np.log(a0) - np.repeat( np.diag(np.log(a0)) , a0.shape[0]      ).reshape(a0.shape)
        theta1 = np.log(a1[:,1]) - np.log(a1[:,0])
        ### HISTOLOGY ###
        # We leave out the diagonal elements, as they are fixed to zero for identifiability reasons.
        # We also leave out the probability of observing 3 given the state is 0 for histology. log(0) would show a warning, but it is ok to continue.

        s=0
        foo = np.delete(theta0[s,:-1],s)
        parameter_vector = np.append(parameter_vector, foo)
        indxs_to = indxs_from + len(foo)
        indxs_from_list.append(indxs_from)
        indxs_to_list.append(indxs_to)
        cons.append({'type': 'ineq',
                     'fun' : lambda x, ind: 1-np.sum(np.exp(x[np.arange(indxs_from_list[ind],indxs_to_list[ind])])),
                     'jac' : lambda x, ind: helper_function(x,lambda y: -np.exp(y),np.arange(indxs_from_list[ind],indxs_to_list[ind]))
                     })
        indxs_from = indxs_to

        for s in xrange(1,nDiseaseStates):
            foo = np.delete(theta0[s,:],s)
            parameter_vector = np.append(parameter_vector, foo)
            indxs_to = indxs_from + len(foo)
            indxs_from_list.append(indxs_from)
            indxs_to_list.append(indxs_to)
            #print 's: ', s
            #print 'foo: ', foo
            #print 'parameter_vector: ', parameter_vector
            #print 'indxs_from: ', indxs_from
            #print 'indxs_to: ', indxs_to

            cons.append({'type': 'ineq',
                         'fun' : lambda x, ind: 1-np.sum(np.exp(x[np.arange(indxs_from_list[ind],indxs_to_list[ind])])),
                         'jac' : lambda x, ind: helper_function(x,lambda y: -np.exp(y),np.arange(indxs_from_list[ind],indxs_to_list[ind]))
                        })
            indxs_from = indxs_to

        ### HPV ###
        s=0
        foo = theta1[s]
        parameter_vector = np.append(parameter_vector, foo)
        indxs_to = indxs_from + 1
        indxs_from_list.append(indxs_from)
        indxs_to_list.append(indxs_to)
        cons.append({'type': 'ineq',
                     'fun' : lambda x, ind: -x[np.arange(indxs_from_list[ind],indxs_to_list[ind])][0],
                     'jac' : lambda x, ind: helper_function(x,lambda y: -y,np.arange(indxs_from_list[ind],indxs_to_list[ind]))
                     })
        indxs_from = indxs_to

        for s in xrange(1,nDiseaseStates):
            foo = theta1[s]
            parameter_vector = np.append(parameter_vector, foo)
            indxs_to = indxs_from + 1
            indxs_from_list.append(indxs_from)
            indxs_to_list.append(indxs_to)
            cons.append({'type': 'ineq',
                         'fun' : lambda x, ind: x[np.arange(indxs_from_list[ind],indxs_to_list[ind])][0],
                         'jac' : lambda x, ind: helper_function(x,lambda y: y, np.arange(indxs_from_list[ind],indxs_to_list[ind]))
                        })
            indxs_from = indxs_to
        #parameter_vector = np.append(parameter_vector, theta1)  # 11-14     hpv
    if "Eta" in ind:
        parameter_vector = np.append(parameter_vector, np.log(Eta.reshape([1,np.prod(Eta.shape)])[0])) # 15-26  eta: test number
        indxs_to = indxs_from + len(np.log(Eta.reshape([1,np.prod(Eta.shape)])[0]))
        indxs_from = indxs_to
    if "W" in ind:
        for W_i in W:
            parameter_vector = np.append(parameter_vector, W_i) #27-60 ?    # age effect intensity coefficient
            indxs_to = indxs_from + len(W_i)
            indxs_from = indxs_to
    if "Gamma" in ind:
        parameter_vector = np.append(parameter_vector, Gamma)
        indxs_to = indxs_from + len(Gamma)
        indxs_from = indxs_to
        #for Gamma_i in Gamma:
        #    parameter_vector = np.append(parameter_vector, Gamma_i) #61-?   # HPV status intensity coefficient
    if "Zeta" in ind:
        parameter_vector = np.append(parameter_vector, Zeta)
        indxs_to = indxs_from + len(Zeta)
        indxs_from = indxs_to
        #for Zeta_i in Zeta:
        #    parameter_vector = np.append(parameter_vector, Zeta_i)   #?-?   # Previously treated intensity coefficient
    # make sure you are setting the correct parameters to zero. Different versions of the code assume different 0 pars.
    # currently the most up-to-date versions set the diagonal elements to zero.
    if "A" in ind:
        #parameter_vector = np.append(parameter_vector, np.delete(A,(nDiseaseStates+1)*np.arange(nDiseaseStates)) ) #A

        for s in xrange(nDiseaseStates):
            foo = np.delete(A[s,:],s)
            parameter_vector = np.append(parameter_vector, foo)
            indxs_to = indxs_from + len(foo)
            indxs_from_list.append(indxs_from)
            indxs_to_list.append(indxs_to)
            cons.append({'type': 'ineq',
                         'fun' : lambda x, ind: 1-np.sum(np.exp(x[np.arange(indxs_from_list[ind],indxs_to_list[ind])])),
                         'jac' : lambda x, ind: helper_function(x,lambda y: -np.exp(y),np.arange(indxs_from_list[ind],indxs_to_list[ind]))
                        })
            indxs_from = indxs_to
    if "B" in ind:
        parameter_vector = np.append(parameter_vector, np.delete(B,(nDiseaseStates+1)*np.arange(nDiseaseStates)) ) #B
        indxs_to = indxs_from + len(np.delete(B,(nDiseaseStates+1)*np.arange(nDiseaseStates)))
        indxs_from = indxs_to
    if "C" in ind:
        for C_i in C[1:]:
            parameter_vector = np.append(parameter_vector, C_i) #C
            indxs_to = indxs_from + len(C_i)
            indxs_from = indxs_to

    return parameter_vector, tuple(cons)    #, [indxs_1,indxs_2,indxs_3]

def ParVec2ParList_simplified(parameter_vector, nDiseaseStates, nTests, nIntensities, nTestOutcomes, nAlpha, nEta, nW, W_num, nGamma, nZeta, nA, nB, nC, ind ):
    nStates = nDiseaseStates
    s = 0
    if "Alpha" in ind:
        #nDiseaseStates
        nTheta = nAlpha - 2*nStates - 1
        Theta = parameter_vector[:nTheta]
        theta0 = Theta[:-nStates] #
        theta0 = np.insert(theta0,2,-np.infty)
        theta0 = np.insert(theta0,nStates*np.arange(nStates), np.zeros(nStates))
        theta0 = theta0.reshape([nStates,nStates])
        theta1 = Theta[-nStates:]
        theta1 = np.insert(theta1,np.arange(nStates),np.zeros(nStates))
        theta1 = theta1.reshape([nStates,2])

        a0 = np.zeros([nStates,nStates])
        a1 = np.zeros([nStates,2])

        for t in xrange(nStates):
            a0[t,:] = np.exp(    theta0[t,:]  - np.log(  np.sum( np.exp(theta0[t,:])   )     )      )
            a1[t,:] = np.exp(    theta1[t,:]  - np.log(  np.sum( np.exp(theta1[t,:])   )     )      )

        Alpha = [a0,a1]
        s = s + nTheta
    else:
        a0 = np.zeros([nStates,nStates])
        a1 = np.zeros([nStates,2])
        Alpha = [a0,a1]

    if "Eta" in ind:
        Eta   = np.exp(parameter_vector[s:(s + nEta)])
        s = s + nEta
        Eta = Eta.reshape([ nStates, nTests  ])
    else:
        print 'EEEEELLLLLSSSSSSEEEEEEEEEE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        Eta = np.zeros([ nStates, nTests  ])
    if "W" in ind:
        W_vec = parameter_vector[s:(s+nW)]
        W = []
        count = 0
        for i in W_num:
            W.append(W_vec[count:(count+i)])
            count += i
        s = s + nW
    else:
        W = [np.zeros(i) for i in W_num]

    #if "Gamma" in ind:
    #    Gamma = parameter_vector[s:(s+nGamma)]
    #    s = s + nGamma
    #else:
    #    Gamma = np.zeros(nGamma)

    if "Gamma" in ind:  # HPV status
        Gamma = parameter_vector[s:(s+nGamma)]
        #Gamma_vec = parameter_vector[s:(s+nGamma)]
        #Gamma = []
        #count = 0
        #for i in Gamma_num:
        #    Gamma.append(Gamma_vec[count:(count+i)])
        #    count += i
        s = s + nGamma
    else:
        Gamma = np.zeros(nGamma)
        #Gamma = [np.zeros(i) for i in Gamma_num]

    if "Zeta" in ind: # previously treated status
        Zeta = parameter_vector[s:(s+nZeta)]
        #Zeta_vec = parameter_vector[s:(s+nZeta)]
        #Zeta = []
        #count = 0
        #for i in Zeta_num:
        #    Zeta.append(Zeta_vec[count:(count+i)])
        #    count += i
        s = s + nZeta
    else:
        Zeta = np.zeros(nZeta)
        #Zeta = [np.zeros(i) for i in Zeta_num]

    if "A" in ind:
        Asub = parameter_vector[s:(s+nA-nStates)]
        A = np.insert(Asub,nStates*np.arange(nStates),np.zeros(nStates)).reshape([nStates,nStates])
        s = s + nA - nStates
    else:
        Asub = np.zeros(nA - nStates)
        A = np.insert(Asub,nStates*np.arange(nStates),np.zeros(nStates)).reshape([nStates,nStates])

    if "B" in ind:
        Bsub = parameter_vector[s:(s+nB-nStates)]
        B = np.insert(Bsub,nStates*np.arange(nStates),np.zeros(nStates)).reshape([nStates,nStates])
        s = s + nB - nStates
    else:
        Bsub = np.zeros(nB - nStates)
        B = np.insert(Bsub,nStates*np.arange(nStates),np.zeros(nStates)).reshape([nStates,nStates])
    # make sure we are inserting at the correct places. This should match up with the inverse function above.
    if "C" in ind:
        C = [np.zeros(C_num[0])]
        for i in range(nStates - 1):
            C.append(parameter_vector[s: s+inv_num])
            s += inv_num
    else:
        C = [np.zeros(C_num[0]) for i in range(nStates)]

    return Alpha, Eta, W, Gamma, Zeta, A, B, C

def Age2Comp(age): # This function is to specify the intensity component for the certain age(value) and certain transition index.  Interval looks like [ ).
    temp = 0
    while age >= inv[temp]:
        temp += 1
    return (temp)

def multinomial_logpmf(n, counts, theta):
    n = int(n)
    counts = counts.astype(int)
    if n != np.sum(counts):
        out = -np.infty
        #print 'n != np.sum(counts)'
        #print 'n: ', n
        #print 'sum(counts):', np.sum(counts)
    elif n==0:
        out = 0
    elif np.prod(theta) == 0:
        if np.sum(counts[theta==0]) > 0:
            out = -np.infty
            #print 'theta has zero and pos obs! '
        else:
            #n = int(np.sum(theta > 0))
            counts = counts[theta > 0]
            theta = theta[theta > 0]
            out = multinomial_logpmf(n,counts,theta)

    else:
        if len(theta) == len(counts):
            out = np.dot(counts,np.log(theta))
        else:
            # This deals with the case of HPV where the only possible outcomes are 0-1.
            # If we want to generalize this to arbitrary numbers of outcomes, we will need to be more careful.
            out = np.dot(counts[:len(theta)], np.log(theta))
        out += np.sum( [ np.log(i+1) for i in xrange(n)   ] )
        for x in counts:
            if x > 0:
                out -= np.sum( [ np.log(i+1) for i in xrange(x)  ]  )
    return out

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
        matrix[i,i] = - np.sum(matrix[i,:]) #+ matrix[i,i]
    try:
        #out1 = expm(dt*matrix)  # too slow!!!
        out = expm2(dt*matrix)  # so far so good...
        #out3 = expm3(dt*matrix) # too unstable!!!
    except:
        print 'lambdas: ', lambdas
        print 'matrix: ', matrix
        print 'dt: ', dt
        print 'dt*Lambda: ', dt*matrix
        print 'exp(dt*Lambda): ', expm2(dt*matrix)
        out  = expm2(dt*matrix)
        #out = np.eye(MPmatrix.shape[0])

    out = np.where(out < 0, 0., out)
    out = np.where(out > 1, 1., out)
    norm = np.repeat(np.sum(out,1),out.shape[0]).reshape(out.shape)
    out = out/norm
    return out

def ProbTransition_4(MPmatrix, W, Gamma, Zeta, start, end, hpv_probs=np.array([0,0,0,0]), hpv_status=0, treatment_status=0, verbose = False):
    '''
        'matrix' should be a square N-by-N matrix of ones and zeros that defines the intensity matrix of the markov process.
        A 1 at element ij indicates a possible transition between states i and j.
        A 0 at element ij means no possible transition between states i and j.

        Because this is a continuous time Markov Process the diagonals are forced to be zero.

        hpv_status is 0,1 or -1. If -1, then status is unknown.
        treatment_status is 0 or 1.

    '''
    if verbose:
        print ("Computation for P starts")
        print ("W = {}, Gamma = {}, Zeta = {}, start = {}, end = {}".format(W, Gamma, Zeta, start, end))
    temp = start

    nStates = MPmatrix.shape[0]
    matrix = np.eye(nStates)

    if hpv_status == -1:
        count = 2
    else:
        count = 0

    temp_M = np.array(MPmatrix)

    for i in range(len(hpv_probs)):
        temp_M[i,:] = hpv_probs[i]*MPmatrix[i,:]

    temp_hpv_probs = temp_M[np.where(MPmatrix > 0)]

    while (temp < end):
        temp_component = Age2Comp(temp)
        end_component = Age2Comp(end)
        if temp_component < end_component:
            dt = (inv[temp_component] - temp) * 12 # back to month
            temp_W = np.array([w[temp_component] for w in W])

            if count == 0:
                temp_W += Gamma*hpv_status
            elif count > 0:
                temp_W += Gamma*temp_hpv_probs

            temp_W += Zeta*treatment_status

            matrix = np.dot(matrix, ProbTransition_interval(MPmatrix, dt, np.exp(temp_W)  ))
            temp = inv[temp_component]
        else:
            dt = (end - temp) * 12 # back to month
            temp_W = np.array([w[temp_component] for w in W])

            if count == 0:
                temp_W += Gamma*hpv_status
            elif count > 0:
                temp_W += Gamma*temp_hpv_probs

            temp_W += Zeta*treatment_status

            matrix = np.dot(matrix, ProbTransition_interval(MPmatrix, dt, np.exp(temp_W)))
            temp = inv[temp_component]
        if verbose:
            print ("temp = {}, temp_component = {}".format(temp, temp_component))
        count += 1
    return(matrix)

def NegativeLogLikelihood_caller(parameter_vector, nStates, nTests, nIntensities, nTestOutcomes, MPmatrix,nAlpha, nEta, nW, nGamma, nZeta, nA, nB, nC, times, testTypes, observations, states, ages, treatment_indx, censor_ages, death_states, stop, simplified = False, ind = ["Alpha", "Eta", "W", "Gamma","Zeta", "A", "B", "C"], cons=[]): # stop specifies if the optimization stops
    ''' This simply evaluates the log loikelihood of the model given ALL data, even latent variables.
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
    if len(cons) > 0:
        do_penalize_constraints = True
    else:
        do_penalize_constraints = False
    first_inf = -1 # This is an indicator used for debug.
    comm.Barrier()
    # t_s = time.clock()
    # print ("rank {}: stop sign {}".format(rank, stop[0]))
    stop[0] = comm.bcast(stop[0], 0)
    summ = 0
    if stop[0] == 0:
        parameter_vector = comm.bcast(parameter_vector, 0)

        Alpha, Eta, W, Gamma, Zeta, A, B, C = ParVec2ParList_simplified(parameter_vector, nStates, nTests,
                                                                           nIntensities, nTestOutcomes, nAlpha, nEta,
                                                                           nW, W_num,
                                                                           nGamma,
                                                                           nZeta,
                                                                           nA, nB, nC,  ind )

        # initial conditions #
        log_likelihood = 0
        patient_counter = 0
        observed_data = zip(times, testTypes, observations, states, ages, treatment_indx, censor_ages, death_states)
        for patient_times, patient_tests, patient_observations, patient_states, patient_ages, patient_treatment_indx, censor_age, death_state  in observed_data:
            if first_inf  < 0 and log_likelihood == -np.infty:
                first_inf = 1
                print '0: ', log_likelihood, 'rank', rank, 'patient_counter', patient_counter
            treatment_status = 0
            # initial state probability
            # log_likelihood += C[patient_states[0]] + D[patient_states[0]]*patient_ages[0] - np.log( np.sum( np.exp(  C[patient_states[0]] + D[patient_states[0]]*patient_ages[0]  ) ) )
            patient_counter = patient_counter + 1

            # Compute the probability of state in the first screening test
            if "C" in ind:
                #log_likelihood += C[patient_states[0]] - np.log( np.sum( np.exp(C) ) )
                patient_age = patient_ages[0]
                age_interval = Age2Comp(patient_age)
                patient_c = [c[age_interval] for c in C]
                log_likelihood += patient_c[patient_states[0]] - np.log( np.sum( np.exp(patient_c) ) )
            # for 1st step debug.
            if first_inf  < 0 and log_likelihood == -np.infty:
                first_inf = 1
                print '1: ', log_likelihood, 'rank', rank, 'patient_counter', patient_counter
                print 'patient_c: ', patient_c
                print 'C: ', C
                print 'patient_age: ', patient_age
                print 'patient_interval: ', age_interval
                print 'patient_state ', patient_states
                print np.exp(patient_c)
                print np.sum( np.exp(patient_c) )
                print np.log( np.sum( np.exp(patient_c) ))
                print patient_c[patient_states[0]] - np.log( np.sum( np.exp(patient_c) ) )
            # initial test type probability

            # Compute the probabilty of screening test times.
            if "Eta" in ind:
                log_likelihood += np.sum(poisson.logpmf(patient_tests[0,:], Eta[patient_states[0],:]))
            # for 2ed step debug
            if first_inf  < 0 and log_likelihood == -np.infty:
                first_inf = 1
                print '2: ', log_likelihood, 'rank', rank, 'patient_counter', patient_counter

            # Compute the probability of results for three tests
            for t in xrange(nTests):
                if t == 0 and ("A" in ind or "B" in ind): # For cytology screening tests
                    if simplified:
                        log_theta = A[patient_states[0],:] - np.log(np.sum(np.exp(A[patient_states[0],:]) ) )
                    else:
                        log_theta = A[patient_states[0],:]+ B[patient_states[0],:]*int(patient_ages[0] > 50) - np.log(  np.sum(    np.exp(A[patient_states[0],:]+ B[patient_states[0],:]*int(patient_ages[0] > 50))    ) )
                    theta = np.exp(log_theta)
                    log_likelihood += fb.multinomial_logpmf(patient_tests[0,t], patient_observations[0,t,:], theta)
                elif "Alpha" in ind: # For histology and HPV screening tests
                    theta = Alpha[t-1][patient_states[0],:]
                    log_likelihood += fb.multinomial_logpmf(patient_tests[0,t], patient_observations[0,t,:], theta)
            # for 3rd step debug
            if first_inf  < 0 and log_likelihood == -np.infty:
                first_inf = 1
                print '3: ', log_likelihood, 'rank', rank, 'patient_counter', patient_counter
            # going forward

            #print 'patient_states: ',
            for j in xrange(1,len(patient_times)):
                # transite prob
                if "W" in ind or "Gamma" in ind or "Zeta" in ind:
                    if j-1 in patient_treatment_indx:
                        treatment_status = 1
                        if "Gamma" in ind:
                            if patient_tests[j-1,-1] > 0 and patient_observations[j-1,-1,1] > 0:
                                hpv_status = 1
                            elif patient_tests[j-1,-1] > 0 and patient_observations[j-1,-1,1] == 0:
                                hpv_status = 0
                            else:
                                hpv_status = -1
                            hpv_probs  = currAlpha[1][:,1]
                        else:
                            hpv_status = 0
                            hpv_probs  = np.zeros(nStates)
                        p_transition = ProbTransition_4(MPmatrix, W, Gamma, Zeta,
                                                        patient_ages[j-1], patient_ages[j],
                                                        hpv_probs = hpv_probs,
                                                        hpv_status = hpv_status,
                                                        treatment_status = treatment_status)
                        # print 'P: ',  p_transition, 'patient_ages', patient_ages[j-1], patient_ages[j]
                        ######################################################################################
                        # This is a treatment state, thus the previous state is KNOWN to be normal, i.e. 0. #
                        ######################################################################################
                        log_likelihood += np.log(p_transition[0,patient_states[j]]) # first stage should be zero. NOT A TYPO!!
                    else:
                        treatment_status = 0
                        if "Gamma" in ind:
                            if patient_tests[j-1,-1] > 0 and patient_observations[j-1,-1,1] > 0:
                                hpv_status = 1
                            elif patient_tests[j-1,-1] > 0 and patient_observations[j-1,-1,1] == 0:
                                hpv_status = 0
                            else:
                                hpv_status = -1
                            hpv_probs  = currAlpha[1][:,1]
                        else:
                            hpv_status = 0
                            hpv_probs  = np.zeros(nStates)
                        p_transition = ProbTransition_4(MPmatrix, W, Gamma, Zeta,
                                                        patient_ages[j-1], patient_ages[j],
                                                        hpv_probs = hpv_probs,
                                                        hpv_status = hpv_status,
                                                        treatment_status = treatment_status)
                        # print 'P: ',  p_transition, 'patient_ages', patient_ages[j-1], patient_ages[j]
                        log_likelihood += np.log(p_transition[patient_states[j-1], patient_states[j]])
                # for 4th step debug
                if first_inf  < 0 and log_likelihood == -np.infty:
                    first_inf = 1
                    print '4: ', log_likelihood, 'rank', rank, 'patient_counter', patient_counter

                # test type prob
                if "Eta" in ind:
                    log_likelihood += np.sum(poisson.logpmf(patient_tests[j,:], Eta[patient_states[j],:]))
                # for 5th step debug
                if first_inf  < 0 and log_likelihood ==  -np.infty:
                    first_inf = 1
                    print '5: ', log_likelihood, 'rank', rank, 'patient_counter', patient_counter

                # observation prob
                for test in xrange(nTests):
                    if test == 0 and ("A" in ind or "B" in ind):
                        if simplified:
                            log_theta = A[patient_states[j],:] - np.log(np.sum(np.exp(A[patient_states[j],:]) ) )
                        else:
                            log_theta = A[patient_states[j],:]+ B[patient_states[j],:]*int(patient_ages[j] > 50) - np.log(  np.sum(    np.exp(A[patient_states[j],:]+ B[patient_states[j],:]*int(patient_ages[j] > 50))    ) )
                        theta = np.exp(log_theta)
                        log_likelihood += fb.multinomial_logpmf(patient_tests[j,test], patient_observations[j,test,:], theta  )
                    elif "Alpha" in ind:
                        theta = Alpha[test-1][patient_states[j],:]
                        log_likelihood += fb.multinomial_logpmf(patient_tests[j,test], patient_observations[j,test,:], theta  )
                    # if rank == 5 and patient_counter == 87:
                    #     print ("A = {}".format(A))
                    #     print ("j = {}, patient_states[j] = {}, test = {}, patient_tests = {}, patient_observations = {}, theta = {}, value = {}".format(j, patient_states[j], test, patient_tests[j, :], patient_observations[j, :, :], theta,fb.multinomial_logpmf(patient_tests[j,test], patient_observations[j,test,:], theta  )))
                    # for 6th step debug
                    if first_inf  < 0 and log_likelihood == -np.infty:
                        first_inf = 1
                        print '6: ', log_likelihood, 'rank', rank, 'patient_counter', patient_counter
                #
            if "W" in ind or "Gamma" in ind or "Zeta" in ind:
                if "Gamma" in ind:
                    if patient_tests[j-1,-1] > 0 and patient_observations[j-1,-1,1] > 0:
                        hpv_status = 1
                    elif patient_tests[j-1,-1] > 0 and patient_observations[j-1,-1,1] == 0:
                        hpv_status = 0
                    else:
                        hpv_status = -1
                    hpv_probs  = currAlpha[1][:,1]
                else:
                    hpv_status = 0
                    hpv_probs  = np.zeros(nStates)
                if censor_age < patient_ages[-1]:
                    # this can happen due to some rounding errors when death is very close to last screening.
                    # Just move the censor date a few month after last visit.
                    censor_age = patient_ages[-1] + 0.25

                p_transition = ProbTransition_4(MPmatrix, W, Gamma, Zeta,
                                                patient_ages[-1], censor_age,
                                                hpv_probs = hpv_probs,
                                                hpv_status = hpv_status,
                                                treatment_status = treatment_status)
                if death_state > 0: # this means censor age is age of 'death', not end of observations.
                    log_likelihood += np.log(p_transition[patient_states[-1],-1])
                else: # this means censor age is age of end of observations, not 'death'. So we know they are still alive at the time the study ended.
                    #log_likelihood += np.log(1. - p_transition[patient_states[-1],-1])
                    p_max_age_transition = ProbTransition_4(MPmatrix, W, Gamma, Zeta,
                                         patient_ages[-1], max_age,
                                         hpv_probs = hpv_probs,
                                         hpv_status = hpv_status,
                                         treatment_status = treatment_status)

                    diff_p =  p_max_age_transition[patient_states[-1],-1] - p_transition[patient_states[-1],-1]
                    if diff_p < 0:
                        print 'patient_ages[-1]: ', patient_ages[-1]
                        print 'censor_age: ', censor_age
                        print 'patient_states[-1]: ', patient_states[-1]
                        print 'p_max_age_transition: ', p_max_age_transition[patient_states[-1],-1]
                        print  'p_transition: ', p_transition[patient_states[-1],-1]
                        print 'diff_p: ', diff_p
                        print 'log(diff_p): ', np.log(diff_p)
                    if diff_p == 0:
                        log_p7 = np.log(10**(-10))
                    else:
                        log_p7 = np.log(p_max_age_transition[patient_states[-1],-1] - p_transition[patient_states[-1],-1])
                    log_likelihood += log_p7
            #for 7th step debug
            if first_inf  < 0 and log_likelihood == -np.infty:
                first_inf = 1
                print '7: ', log_likelihood, 'rank', rank, 'patient_counter', patient_counter
                print 'ages: ', patient_ages
                print 'censor_age: ', censor_age
                print 'death state: ', death_state
                print 'hpv_status: ', hpv_status
                print 'treatment_status: ', treatment_status
                print 'state: ', patient_states[-1]
                print 'P: ', p_transition
                print 'P[s,-1]: ', p_transition[patient_states[-1],-1]
                print 'log(P[s,-1]): ', np.log(p_transition[patient_states[-1],-1])

        summ = comm.reduce(log_likelihood, op = MPI.SUM, root = 0)
        # print ('NegativeLogLikelihood for rank {} is {}'.format(rank, -log_likelihood))
        # print ("time for negative log likelihood computation of rank {} is {}".format(rank, time.clock() - t_s))
    if do_penalize_constraints:
        constraint_functions_evaluated = [dic['fun'](parameter_vector, ind) for dic, ind in zip(cons, xrange(len(cons)))]
        # print constraint_functions_evaluated
        constraint_penalties = [ 1e6*int(g < 0)*g**2 for g in constraint_functions_evaluated  ]
        penalty = np.sum(constraint_penalties)
    else:
        penalty = 0

    if rank == 0:
        return -summ
    else:
        return 0

def Best_Patient_States_treatment(MPmatrix, patient_times, patient_tests, patient_observations, patient_age, patient_treatment_indx, nStates, nTests, currA, currB, currC, currC_treated, currAlpha, currEta, currW, currGamma, currZeta, ind, verbose = False):
    if len(patient_treatment_indx) == 1:
        j = patient_treatment_indx[0]
        bs0 = Best_Patient_States0(MPmatrix, patient_times[:(j+1)], patient_tests[:(j+1)], patient_observations[:(j+1)], patient_age[:(j+1)], 0,
                                   nStates, nTests, currA, currB, currC, currAlpha, currEta, currW, currGamma, currZeta,  ind, verbose)
        bs1 = Best_Patient_States0(MPmatrix, patient_times[j:], patient_tests[j:], patient_observations[j:], patient_age[j:], 1,
                                   nStates, nTests, currA, currB, currC_treated, currAlpha, currEta, currW, currGamma, currZeta, ind, verbose)
        best_states = np.concatenate([bs0,bs1[1:]])

    elif len(patient_treatment_indx) > 1:
        best_states_list = []
        j = patient_treatment_indx[0]
        bs0 = Best_Patient_States0(MPmatrix, patient_times[:(j+1)], patient_tests[:(j+1)], patient_observations[:(j+1)], patient_age[:(j+1)], 0,
                        nStates, nTests, currA, currB, currC, currAlpha, currEta, currW, currGamma, currZeta, ind, verbose)
        best_states_list.append(bs0)
        for i in xrange(len(patient_treatment_indx)-1):
            j = patient_treatment_indx[i]
            k = patient_treatment_indx[i+1] + 1
            bsi = Best_Patient_States0(MPmatrix, patient_times[j:k], patient_tests[j:k], patient_observations[j:k], patient_age[j:k], 1,
                        nStates, nTests, currA, currB, currC_treated, currAlpha, currEta, currW, currGamma, currZeta, ind, verbose)
            best_states_list.append(bsi[1:])
        j = patient_treatment_indx[-1]
        bs1 = Best_Patient_States0(MPmatrix, patient_times[j:], patient_tests[j:], patient_observations[j:], patient_age[j:], 1,
                        nStates, nTests, currA, currB, currC_treated, currAlpha, currEta, currW, currGamma, currZeta, ind, verbose)
        best_states_list.append(bs1[1:])
        best_states = np.concatenate(best_states_list)
    else:
        best_states = Best_Patient_States0(MPmatrix, patient_times, patient_tests, patient_observations, patient_age, 0,
                        nStates, nTests, currA, currB, currC, currAlpha, currEta, currW, currGamma, currZeta, ind, verbose)
    #########
    # check #
    #########
    if len(best_states) != len(patient_times):
        print 'Error in Best_Patient_States_treatment: best state vector length does not match patient_times vector length.'

    return best_states

def Best_Patient_States0(MPmatrix, patient_times, patient_tests, patient_observations, patient_age, patient_treatment_status, nDiseaseStates, nTests, currA, currB, currC, currAlpha, currEta, currW, currGamma, currZeta, ind, verbose = False):   ####verbose is used for debug.
    nStates = nDiseaseStates

    ### Forward Pass ###
    P_forward_matrices = [ np.zeros([nStates, nStates]) for t in patient_times]
    #
    # initial conditions #
    #log_ell = currC + currD*patient_age[0] - np.log( np.sum( np.exp(  currC + currD*patient_age[0]  ) ) )
    age_interval = Age2Comp(patient_age[0])
    patient_c = [c[age_interval] for c in currC]
    log_ell = patient_c - np.log( np.sum( np.exp( patient_c  ) ) )
    #

    for s in xrange(nStates):
        if "Eta" in ind:
            log_ell[s] += np.sum(poisson.logpmf(patient_tests[0,:], currEta[s,:]))
        #
        for t in xrange(nTests):
            if t == 0:
                log_theta = currA[s,:]+ currB[s,:]*int(patient_age[0] > 50) - np.log(  np.sum(    np.exp(currA[s,:]+ currB[s,:]*int(patient_age[0] > 50))    ) )
                theta = np.exp(log_theta)
            else:
                theta = currAlpha[t-1][s,:]
            #
            log_ell[s] += multinomial_logpmf(patient_tests[0,t], patient_observations[0,t,:], theta  )

    log_L = np.log( np.sum( np.exp( log_ell  )  ) )
    log_Q = log_ell - log_L
    Q1 = Q = np.exp(log_Q)
    #Z  = np.max( log_ell )
    if verbose:
        print("P(S_1 = s*|T_1, R_1) = {}".format(Q))

    # go forward
    for j in xrange(1,len(patient_times)):
        if "Gamma" in ind:
            if patient_tests[j-1,-1] > 0 and patient_observations[j-1,-1,1] > 0:
                hpv_status = 1
            elif patient_tests[j-1,-1] > 0 and patient_observations[j-1,-1,1] == 0:
                hpv_status = 0
            else:
                hpv_status = -1
            hpv_probs  = currAlpha[1][:,1]
        else:
            hpv_status = 0
            hpv_probs  = np.zeros(nStates)
        p_transition = ProbTransition_4(MPmatrix, currW, currGamma, currZeta,
                                        patient_age[j-1], patient_age[j],
                                        hpv_probs = hpv_probs,
                                        hpv_status = hpv_status,
                                        treatment_status = patient_treatment_status)
        log_prob_obs = np.zeros(nStates)
        for s in xrange(nStates):
            if "Eta" in ind:
                log_prob_obs[s] += np.sum(poisson.logpmf(patient_tests[j,:], currEta[s,:]))
            for test in xrange(nTests):
                if test == 0:
                    log_theta = currA[s,:]+ currB[s,:]*int(patient_age[j] > 50) - np.log(  np.sum(    np.exp(currA[s,:]+ currB[s,:]*int(patient_age[j] > 50))    ) )
                    theta = np.exp(log_theta)
                else:
                    theta = currAlpha[test-1][s,:]
                log_prob_obs[s] += multinomial_logpmf(patient_tests[j,test], patient_observations[j,test,:], theta  )

        log_proportional_prob = np.repeat(log_Q,nStates).reshape([nStates,nStates]) + np.transpose(np.repeat(log_prob_obs,nStates).reshape([nStates,nStates])) + np.log(p_transition[:nStates,:nStates])
        proportional_prob = np.exp(log_proportional_prob)
        #
        P_forward_matrices[j] = np.exp( log_proportional_prob - np.log(np.sum(proportional_prob) ) )
        #
        Q = np.sum(P_forward_matrices[j],0)
        log_Q = np.log(Q)

    if verbose:
        print("Forward Matrices: ", P_forward_matrices)

    #####################
    ### Backward Pass ###
    #####################
    P_backward_matrices = [ np.zeros([nStates, nStates]) for t in patient_times]
    N = len(patient_times)
    best_states = np.zeros(N).astype(int)
    # go backward
    P_backward_matrices[-1] = P_forward_matrices[-1]
    # q_backward = np.sum(P_backward_matrices[-1], 1)
    if N == 1:
        best_states[0] = np.argmax(Q1)
    else:
        maxindex = np.unravel_index(P_backward_matrices[-1].argmax(),P_backward_matrices[-1].shape)
        best_states[-1] = maxindex[1]
        best_states[-2] = maxindex[0]
    for j in xrange(2,N):
        # q_forward  = np.sum(P_forward_matrices[-j],0)
        # # if verbose:
        #     # print ("q_foward: {}, P_forward_matrix[-j]: {}".format(q_forward, P_forward_matrices[-j]))
        #     # print ("1. ", np.log(P_forward_matrices[-j]))
        #     # print ("2. ", np.log(q_backward))
        #     # print ("3. ", np.log(q_forward))
        #     # print ("4. ", np.transpose(np.repeat(np.log(q_backward) - np.log(q_forward),nStates).reshape([nStates,nStates])))
        # P_backward_matrices[-j] = np.exp(np.log(P_forward_matrices[-j]) + np.transpose(np.repeat(np.log(q_backward) - np.log(q_forward),nStates).reshape([nStates,nStates])))
        # mask = np.isnan(P_backward_matrices[-j]) ###
        # P_backward_matrices[-j][mask] = 0 ###
        # q_backward = np.sum(P_backward_matrices[-j],1)
        best_states[-j-1] = np.argmax(P_forward_matrices[-j][:, best_states[-j]])
    if verbose:
        print ("Backward matrices: ", P_backward_matrices)
        print ("best_states: ", best_states)

    return best_states

def approx_fprime(f, x0, epsilon, *args):
    # print ('start to compute f0')
    f0 = f(x0, *args)
    # print ('Got f0!')
    res = np.zeros_like(x0)
    for j in range(len(x0)):
        # print('x0:', x0)
        xj = np.copy(x0)
        # print('xj =: ', xj)
        xj[j] += epsilon
        fp_j = (f(xj, *args) - f0)/epsilon
        res[j] = fp_j
        print ('{}th element of derivatives has been computed!'.format(j))
    return res

def approx_fprime_j(f, x0, j, epsilon, *args): ### f'_j
    f0 = f(x0, *args)
    xj = np.copy(x0)
    xj[j] += epsilon
    fp_j = (f(xj, *args) - f0)/epsilon
    print ('{}th element of derivatives has been computed!'.format(j))
    return fp_j

def Hessian_computation(f, x0, epsilon, diag, *args):
    n = x0.shape[0]
    print("f1 computation starts!")
    f1 = approx_fprime(f, x0, epsilon, *args)
    print("f1 has been computed!")
    if diag:
        hessian = np.zeros(n)
    else:
        hessian = np.zeros((n,n))
    # The next loop fill in the matrix
    xx = x0
    for j in xrange(n):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        if diag:
            # print (j)
            # print (f1)
            # print (hessian)
            # print (approx_fprime_j(f, x0, j, epsilon, *args))
            hessian[j] = (approx_fprime_j(f, x0, j, epsilon, *args) - f1[j])/epsilon
        else:
            f2 = approx_fprime(f, x0, epsilon, *args)
            hessian[:, j] = (f2 - f1)/epsilon # scale...
        ###
        xx[j] = xx0 # Restore initial value of x0
    return hessian


### Import data ###
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print 'rank', rank
    print 'size', size
    nn = sys.argv[1]
    if len(sys.argv) > 2:
        min_age = int(sys.argv[2])
        nn += '_' + sys.argv[2]
    else:
        min_age = 16
    if min_age > 16:
        do_truncate_ages = True
    else:
        do_truncate_ages = False

    nonzero_data = True # Default choice
    if len(sys.argv) > 3:
        nn += '_' + sys.argv[3]
        if sys.argv[3] == 'nonzero':
            nonzero_data = True
        else:
            nonzero_data = False

    do_penalize_constraints = True

    ### Initialzation
    nStates = 4
    nTests = 3
    n_patients_per_proc = 100
    ind = ["Alpha", "Eta", "W", "A", "C"]

    do_GetCurrentValues = True
    diag = True

    # Location for results 
    save_location = "../model_validation/"

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    current_location = "~/"

    # Age breakpoints 
    inv = [16, 19, 22, 25, 30, 35, 40, 45, 50, 55, 200]
    inv_num = len(inv)

    W_num = [inv_num for i in range(9)] # number of intervals for 9 transition intensities
    C_num = [inv_num for i in range(nStates)]  # number of intervals for C w.r.t. 4 states

    ###################
    #### LOAD DATA ####
    ###################
    data_location = "../distributed_updated_nonzero_data/" # Default dataset is nonzero
    subset_data_location = data_location + 'p%s/'%(str(rank))

    # load data
    times           =  pickle.load(open( subset_data_location + 'mcmcPatientTimes', 'r'))          #
    testTypes       =  pickle.load(open( subset_data_location + 'mcmcPatientTestTypes', 'r'))      #
    observations    =  pickle.load(open( subset_data_location + 'mcmcPatientObservations', 'r'))   #
    regressors      =  pickle.load(open( subset_data_location + 'mcmcPatientRegressors', 'r'))     #
    hasCancer       =  pickle.load(open( subset_data_location + 'mcmcPatientHasCancer', 'r'))      #
    firstHist3      =  pickle.load(open( subset_data_location + 'mcmcPatientFirstHist3', 'r'))     #
    hyster_indx     =  pickle.load(open( subset_data_location + 'mcmcPatientHysterIndx', 'r'))     #
    treatment_indx  =  pickle.load(open( subset_data_location + 'mcmcPatientTreatmentIndx', 'r'))           #
    cancerconf_indx =  pickle.load(open( subset_data_location + 'mcmcPatientCancerConfirmationIndx', 'r'))  #
    censor_ages     =  pickle.load(open( subset_data_location + 'mcmcPatientCensorDates', 'r'))             #
    death_states    =  pickle.load(open( subset_data_location + 'mcmcPatientDeathStates', 'r'))             #

    nPatients = len(times) # number of Patients
    nVisits =  [ len(t) for t in times ] # the number of screening tests for each patient
    # define Markov Process topology with MPmatrix. The diagonal should be zeros.
    # A one in element (i,j) indicates a possible transition between states i and j.
    MPmatrix = np.zeros([nStates+1,nStates+1])
    MPmatrix[0,1] = 1
    MPmatrix[1,0] = 1
    MPmatrix[1,2] = 1
    MPmatrix[2,1] = 1
    MPmatrix[2,3] = 1
    MPmatrix[:-1,-1] = 1
    #
    nIntensities = np.sum(MPmatrix).astype(int)
    #
    temp_ages = regressors[1]
    ages = []
    # Reset age
    for temp_patient_ages, patient_times in zip(temp_ages, times):
        new_patient_ages = temp_patient_ages[0] + patient_times/12.0
        ages.append(new_patient_ages)
    #print (ages)

    times           =  times[:n_patients_per_proc]
    testTypes       =  testTypes[:n_patients_per_proc]
    observations    =  observations[:n_patients_per_proc]
    ages            =  ages[:n_patients_per_proc]
    hasCancer       =  hasCancer[:n_patients_per_proc]
    firstHist3      =  firstHist3[:n_patients_per_proc]
    hyster_indx     =  hyster_indx[:n_patients_per_proc]
    treatment_indx  =  treatment_indx[:n_patients_per_proc]
    cancerconf_indx =  cancerconf_indx[:n_patients_per_proc]
    censor_ages     =  censor_ages[:n_patients_per_proc]
    death_states    =  death_states[:n_patients_per_proc]

    max_age = np.max([100, np.max(censor_ages) + 1 ])


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
                del hasCancer[p]
                del firstHist3[p]
                del hyster_indx[p]
                del treatment_indx[p]
                del cancerconf_indx[p]
                del censor_ages[p]
                del death_states[p]

    print 'original n patients: ', nPatients
    print 'Length of times: ', len(times)

    # After the censoring, we reset the nPatients and nVisits
    nPatients = len(times)
    nVisits = [ len(t) for t in times ]

    nTestOutcomes = np.array([nStates, nStates, 2]) # number of different outcomes for three different tests
    currStates = [ np.zeros(nvs).astype(int) for nvs in nVisits  ]

    if do_GetCurrentValues:
           # currPars = getValues.GetCurrentValues_2(current_location)
        estPars = pickle.load(open('mcmc_starting_parameters','r'))
        estAlpha  = estPars[2] #
        estEta    = estPars[3]
        estW      = estPars[4]
        estGamma  = estPars[5]
        estZeta = estPars[6]
        estA      = estPars[7]
        estB      = estPars[8]
        estC      = estPars[9]
        estC_treated = [np.zeros(C_num[0]), -np.infty*np.ones(C_num[1]), -np.infty*np.ones(C_num[2]), -np.infty*np.ones(C_num[3]) ]
    else:
        print ("Check EM estimates please!") 

    nAlpha =  np.sum([np.prod(a.shape) for a in estAlpha])
    nEta = np.prod(estEta.shape)
    nW = np.sum([len(a) for a in estW])
    nGamma = len(estGamma)
    nZeta = len(estZeta)
    nA = np.prod(estA.shape)
    nB = np.prod(estB.shape)
    nC = np.sum(len(a) for a in estC)

    est_parameter_vector, cons = ParList2ParVec_simplified(estAlpha, estEta, estW, estGamma, estZeta, estA, estB, estC, ind, nStates)
    true_nPars = len(est_parameter_vector)
    simplified = True

    ##############################
    ### Compute current states ###
    ##############################
    currStates = []
    for patient_times, patient_tests, patient_observations, patient_ages, patient_treatment_indx in zip(times, testTypes, observations, ages, treatment_indx):
        patient_states = Best_Patient_States_treatment(MPmatrix, patient_times, patient_tests, patient_observations, patient_ages, patient_treatment_indx, nStates, nTests, estA, estB, estC, estC_treated, estAlpha, estEta, estW, estGamma, estZeta, ind)
        currStates.append(patient_states)
    comm.Barrier()
    # print (currStates)

    ############
    ### test ###
    ############
    # stop = [0]
    # temp = NegativeLogLikelihood_caller(est_parameter_vector, nStates, nTests, nIntensities, nTestOutcomes, MPmatrix, nAlpha, nEta, nW, nGamma, nZeta, nA, nB, nC, times, testTypes, observations, currStates, ages, treatment_indx, censor_ages, death_states, stop, simplified, ind, cons)
    
    ################################################
    ### Summarize the time interval information ####
    ################################################
    # Summarize the screening test intervals
    time_intervals = []
    for patient_ages in ages:
        time_intervals += np.diff(patient_ages).tolist()
    print ("rank {} has {} time invervals".format(rank, len(time_intervals)))
    comm.Barrier()
    total_time_intervals = comm.gather (time_intervals, root = 0)
    # if rank == 0:
    #     print("size: {}".format(size))
    #     total_time_intervals = [time_interval for time_intervals in total_time_intervals for time_interval in time_intervals]
    #     print("The number of total time intervals is {}.".format(len(total_time_intervals)))
    #     plt.figure()
    #     plt.title("The boxplot for intervals")
    #     plt.boxplot(total_time_intervals)
    #     plt.savefig('boxplot_invs.png')
    #     plt.show()

    O = np.zeros([11, 10, 5, 5])
    E = np.zeros([11, 10, 5, 5])
    hpv_status = 0
    hpv_probs = np.zeros(nStates)
    ### Compute the observed transition number and expected transition number ###
    for patient_ages, patient_states, patient_treatment_indx in zip(ages, currStates, treatment_indx):
        count = 0
        for patient_age, patient_state in zip(patient_ages, patient_states):
            if count == 0:
                prev_patient_age = patient_age
                prev_patient_comp = Age2Comp(patient_age)
                if count in patient_treatment_indx:
                    prev_patient_state = 0
                    treatment_status = 1
                else:
                    prev_patient_state = patient_state
                    treatment_status = 0
            else:
                patient_comp = Age2Comp(patient_age)
                # print (patient_comp)
                if patient_comp == prev_patient_comp:
                    patient_age_diff = patient_age - prev_patient_age
                    # print(patient_age_diff)
                    if patient_age_diff < 3:
                        # print (patient_comp,  int(patient_age_diff/0.3), prev_patient_state, patient_state)
                        O[patient_comp, int(patient_age_diff/0.3), prev_patient_state, patient_state] += 1
                        p_transition = ProbTransition_4(MPmatrix, estW, estGamma, estZeta,
                                                        prev_patient_age, patient_age,
                                                        hpv_probs = hpv_probs,
                                                        hpv_status = hpv_status,
                                                        treatment_status = treatment_status)
                        for r in xrange(5):
                            for s in xrange(5):
                                E[patient_comp, int(patient_age_diff/0.3), r, s] += p_transition[r, s]
                prev_patient_age = patient_age
                prev_patient_comp = Age2Comp(patient_age)
                if count in patient_treatment_indx:
                    prev_patient_state = 0
             cd cd       treatment_indx = 1
                else:
                    prev_patient_state = patient_state
                    treatment_indx = 0
            count += 1

    ### Compute the goodness of fit test statistics ###
    comm.Barrier()
    O = comm.reduce(O, op = MPI.SUM, root = 0)
    E = comm.reduce(E, op = MPI.SUM, root = 0)
    if rank == 0:
        O = np.array(O)
        E = np.array(E)
        # print (O)
        # print (E)
        T = np.nansum((O-E)**2/E)
        print('T value is {}'.format(T))
        df = 1770
        print(stats.chi2.cdf(T, df))
        p_value = 1 - stats.chi2.cdf(T, df)
        print('P value is {}'.format(p_value))
        with open(save_location+'T_and_p_value'+nn, 'wb') as t_res:
            pickle.dump([T, p_value], t_res)
