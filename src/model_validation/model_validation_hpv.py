'''

Regression for HPV with respect to cytolopy, histology resutls and age

'''

import sys
import numpy as np
import pickle
from mpi4py import MPI
import scipy.optimize as optimize

def Logistic(x):
    return (np.exp(x)/(1 + np.exp(x)))

def NegLogLikelihood_caller(beta, X, y, t, stop):
    comm.Barrier()
    stop[0] = comm.bcast(stop[0], 0)
    # print ("1", stop)
    if not stop[0]:
        beta = comm.bcast(beta, 0)
        # print ("current beta = {}".format(beta))
        mu = np.dot(X, beta)
        p = Logistic(mu)
        # print ("current p = {}".format(p))
        res = -sum(y*np.log(p) + (t-y)*np.log(1-p))
        # print (res, rank)
        comm.Barrier()
        summ = comm.reduce(res, op = MPI.SUM, root = 0)
    else:
        summ = 0
    comm.Barrier()
    return (summ)

def NegLogLikelihood_der(beta, X, y, t):
    beta = comm.bcast(beta, 0)
    mu = np.dot(X, beta)
    coef = o - (t-o)*np.exp(mu)/(1 + np.exp(mu))
    res = -np.dot(coef, X)
    return(res)

def NegLoglikelihood_hess(beta, X, y, t):
    beta = comm.bcast(beta, 0)
    mu = np.dot(X, beta)
    res = X.T.dot(diag(t*np.exp(mu)/(1+np.exp(mu)))).dot(X)
    return(res)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nn = sys.argv[1]

    ###################
    #### Load Data ####
    ###################
    data_location = '../distributed_updated_nonzero_data/'
    subdata_location = data_location + 'p%s/'%(str(rank))

    ID              =  pickle.load(open(subdata_location + 'mcmcPatientCancerConfirmationIndx', 'r')) 
    times           =  pickle.load(open(subdata_location + 'mcmcPatientTimes', 'r'))
    regressors      =  pickle.load(open(subdata_location + 'mcmcPatientRegressors', 'r'))
    testTypes       =  pickle.load(open(subdata_location + 'mcmcPatientTestTypes', 'r'))
    observations    =  pickle.load(open(subdata_location + 'mcmcPatientObservations', 'r'))
    treatment_indx  =  pickle.load(open(subdata_location + 'mcmcPatientTreatmentIndx', 'r'))   
    temp_ages = regressors[1]
    ages = []
    # Reset age
    for temp_patient_ages, patient_times in zip(temp_ages, times):
        new_patient_ages = temp_patient_ages[0] + patient_times/12.0
        ages.append(new_patient_ages)

    # The number of patients that one processor runs
    n_patients_per_proc = 100

    ID              =  ID[:n_patients_per_proc]
    testTypes       =  testTypes[:n_patients_per_proc]
    observations    =  observations[:n_patients_per_proc]
    ages            =  ages[:n_patients_per_proc]

    #################################
    #### Select training dataset ####
    #################################
    training_ID = []
    training_testTypes = []
    training_observations = []
    training_ages = []
    training_x = []
    training_y = []
    training_t = []
    for patient_ID, patient_testTypes, patient_observations, patient_ages in zip(ID, testTypes, observations, ages):
        for patient_testType, patient_observation, patient_age in zip(patient_testTypes, patient_observations, patient_ages):
            if patient_testType[2] > 0:
                training_ID.append(patient_ID)
                training_testTypes.append(patient_testType)
                training_observations.append(patient_observation)
                training_ages.append(patient_age)
                training_t.append(patient_testType[2])
                training_y.append(patient_observation[2,1])
                training_x.append(np.array([1] + list(patient_observation[0,:]) + list(patient_observation[1,:]) + [patient_age]))
    training_x = np.array(training_x)
    training_y = np.array(training_y)
    training_t = np.array(training_t)
    print("Training dataset has already selected for rank {}.".format(rank))
    comm.Barrier()
    #####################
    #### Train model ####
    #####################
    init_parameters = np.zeros(training_x.shape[1])
    stop = [0]
    if rank == 0:
        print ("Optimization start!")
        res = optimize.minimize(NegLogLikelihood_caller, x0 = init_parameters, args = (training_x, training_y, training_t, stop), 
                    jac = None, method = "Nelder-Mead", options = {'disp': True})
        stop = [1]

        temp = NegLogLikelihood_caller(init_parameters, training_x, training_y, training_t, stop)

    else:
        while not stop[0]:
            # print ("function called!")
            temp = NegLogLikelihood_caller(init_parameters, training_x, training_y, training_t, stop)
            # print (stop[0])
    if rank == 0:
        print("Model training completed.")
    comm.Barrier()

    if rank == 0:
        est_beta = res.x
    else:
        est_beta = None
    est_beta = comm.bcast(est_beta, 0)
    # print est_beta

    ##########################
    #### Model Validation ####
    ##########################
    confussion_matrix = np.array([[0,0], [0,0]])
    for patient_testTypes, patient_observations, patient_ages in zip(testTypes, observations, ages):
        patient_hpvs = []
        for patient_testType, patient_observation, patient_age in zip(patient_testTypes, patient_observations, patient_ages):        
            if patient_testType[2] > 0:
                hpv_observation = patient_observation[2,0:2]
                patient_hpv = np.argmax(hpv_observation)
                temp_x = np.array([1] + list(patient_observation[0,:]) + list(patient_observation[1,:]) + [patient_age])
                hpv_prob = Logistic(np.dot(temp_x.T, est_beta))
                patient_pred_hpv = int(hpv_prob > 0.5)
                confussion_matrix[patient_hpv, patient_pred_hpv] += 1
    total_confussion_matrix = comm.reduce(confussion_matrix, op = MPI.SUM, root = 0)
    if rank == 0:
        TP = total_confussion_matrix[0,0]
        FP = total_confussion_matrix[1,0]
        FN = total_confussion_matrix[0,1]
        TN = total_confussion_matrix[1,1]
        precision = float(TP)/(TP+FP)
        recall = float(TP)/(TP+FN)
        # print(precision, recall)
        with open("hpv_model_validation", "wb") as MV_res:
            pickle.dump([total_confussion_matrix, precision, recall], MV_res)