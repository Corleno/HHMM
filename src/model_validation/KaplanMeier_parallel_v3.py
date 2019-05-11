#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:54:01 2018

This is parallel version of KaplanMeier.py
 
TO DO:  Write other observation patterns and redo all KP estimates. Made a change to KP stat. 

Compute modified Kaplan-Meier statistic.
This requires the scripts 'HMM_simulation_v2.py' and 'HMM_Screening_Data.py' (or the parallel versions) to have been run. 

@author: soper3
"""



import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys 
import os

sys.path.append('../')
sys.path.append('~/')
home = os.path.expanduser('~')   


def KP_stat(referrence_age_range, ages, times, observations_summary):
    inter_event_times = []
    max_inter_event_times = []
    
    event_times  = []
    censor_flags = []
      

    for a,o,t in zip(ages, observations_summary,times):
        age_indices = np.array(a) > referrence_age_range[0]
        a_sub = np.array(a)[ age_indices ]  
        o_sub = np.array(o)[ age_indices ] 
        t_sub = np.array(t)[ age_indices ] 
 
        if len(a_sub) > 0:
            if a_sub[0] >= referrence_age_range[0] and a_sub[0] < referrence_age_range[1]:
                inter_event_times.append([])
                max_inter_event_times.append(np.max(t_sub))
            
                normal_indx   = np.where(np.array(o_sub) ==  0)[0]
                highgrade_indx = np.where(np.array(o_sub) ==  2)[0] # so far this is only used in obs8.
                event_indx    = np.where(np.array(o_sub) ==  1)[0]
                
    
                if len(normal_indx) == 0: # if there are no normal test results we can't use this patient for computig KP-like estimator.
                    current_indx = len(t_sub) # don't got into the while loop!
                else:                
                    current_indx = normal_indx[0] # start at the first normal observation.
            
                while current_indx < len(t_sub)-1:
                    future_events = event_indx[ event_indx > current_indx]
                    if len(highgrade_indx) > 0:
                        future_highgrades = highgrade_indx[  highgrade_indx > current_indx ] 
                    else:
                        future_highgrades = []
                    if len(future_events) > 0: # if there are future events, get the time-to-event.
                        next_event_indx = future_events[0]
                        if len(future_highgrades)>0 and future_highgrades[0] < future_events[0]: # if there is a highgrade event before the event, don't use this.
                            pass
                        else:
                            number_of_months = float(t_sub[next_event_indx]) - float(t_sub[current_indx])
                            inter_event_times[-1].append(number_of_months)
                            event_times.append(number_of_months)
                            censor_flags.append(0)
                        
                        # get all future normals after this event. 
                        future_normals = normal_indx[ normal_indx > next_event_indx  ]
                        if len(future_normals) == 0: # again, if no more normals, then we are done here. 
                            current_indx = len(t_sub)
                        else:
                            current_indx = future_normals[0]  
                    else: # if there are no events, consider it censored and get out of the loop.
                        number_of_months = float(t_sub[-1]) - float(t_sub[current_indx])
                        event_times.append(number_of_months)
                        censor_flags.append(1)
                        current_indx = len(t_sub)
    
    reduced_inter_event_times  = []
    for i in inter_event_times:
        reduced_inter_event_times += i

    unique_inter_event_times = list(np.sort(np.unique(reduced_inter_event_times)))
    KP_vector = np.zeros(len(unique_inter_event_times)+1)
    KP = 1
    KP_vector[0] = KP
    i = 1
    for t_i in unique_inter_event_times:
        #print 't_i: ', t_i
        #n_i = np.sum( (np.array(max_inter_event_times) > t_i).astype(int) )
        n_i = np.sum( (np.array(event_times) > t_i).astype(int) )
        #print 'n_i: ', n_i
        #d_i = len(np.where(reduced_inter_event_times == t_i)[0])
        d_i = len(np.where(np.array(event_times)*(1-np.array(censor_flags)) == t_i)[0] ) # this should be the same as d_i
        #print 'd_i: ', d_i
        
        if n_i == 0:
            KP *= 0.
        else:
            KP *= (1. - (1.*d_i)/n_i)
        #print 'KP: ', KP
        #if (1- (1.*d_i)/n_i) == 0:
        #    print '##################################################'
        KP_vector[i] = KP
        i+=1
        
    return unique_inter_event_times, KP_vector
    
# These functions define what a failure time is for the KP estimator.

def Obs_summary(observations, obs):

    
    ### Use These ###
    if obs == 2:
        return Obs_summary_2(observations), 2    # if cytology and/or histolgoy >= 2 then that is a failure, otherwise it is not. Ignore HPV.
    elif obs == 3:
        return Obs_summary_3(observations), 3    # failure measured from observed 1 to observed 2. Ignore HPV.
    elif obs == 4:
        return Obs_summary_4(observations), 4    # failure measured from observed 0 to observed 2.  Ignore HPV.
    elif obs == 5:
        return Obs_summary_5(observations), 5    # failure measured from observed 0 to observed (1,2,3).  Ignore HPV.
    elif obs == 6:
        return Obs_summary_6(observations), 6    # failure measured from observed 0 to observed 3.  Ignore HPV.    
    elif obs == 7:
        return Obs_summary_7(observations), 7    # failure measured from observed 0 to observed 1.  Ignore HPV.
    elif obs == 8:
        return Obs_summary_8(observations), 8    # failure measured from observed 1 to observed 0.  Ignore HPV.
    elif obs == 9:
        return Obs_summary_9(observations), 9    # failure measured from observed 1 to observed 0.  Ignore HPV.
    elif obs == 10:
        return Obs_summary_10(observations), 10   # failure measured from observed 1 to observed 0.  Ignore HPV.
    elif obs == 11:
        return Obs_summary_11(observations), 11   # failure measured from observed 1 to observed 0.  Ignore HPV.
    elif obs == 12:
        return Obs_summary_12(observations), 12   # failure measured from observed 1 to observed 0.  Ignore HPV.
    elif obs == 13:
        return Obs_summary_13(observations), 13   # failure measured from observed 1 to observed 0.  Ignore HPV.
    elif obs == 14:
        return Obs_summary_14(observations), 14   # failure measured from observed 1 to observed 0.  Ignore HPV.
  







    else:
        print 'Error: No observation pattern given.'



    # Do Not Use These #
    #return Obs_summary_0(observations), 0    # positive HPV, hist/cyt 2 or hist/cyt 3 is a failure.
    #return Obs_summary_1(observations), 1    # any non-normal test result is a failure.
    
    


# not using this one #
def Obs_summary_0(observations):  # positive HPV, hist/cyt 2 or hist/cyt 3 is a failure.
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if r[2,1] > 0: # if you get a positive HPV test, then that is a failure
                observations_summary[-1].append(1)
            elif np.sum(r[:1,2:]) > 0: # if cytology or histolgoy are 2 or 3, then that is a failure
                observations_summary[-1].append(1)
            else: # otherwise it is not a failure.  
                observations_summary[-1].append(0)
    return observations_summary

# not using this one #
def Obs_summary_1(observations): # any non-normal test result is a failure.
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if r[2,1] > 0: # if you get a positive HPV test, then that is a failure
                observations_summary[-1].append(1)
            elif np.sum(r[:1,1:]) > 0: # if cytology or histolgoy are 1, 2 or 3, then that is a failure
                observations_summary[-1].append(1)
            else: # otherwise it is not a failure.  
                observations_summary[-1].append(0)
    return observations_summary

# using this one #
def Obs_summary_2(observations): # if cytology and/or histolgoy >= 2 then that is a failure. Ignore HPV.
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if np.sum(r[:1,2:]) > 0: # if cytology and/or histolgoy >= 2 (other than death) then that is a failure. 
                observations_summary[-1].append(1)
            elif np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: # normal cytology of histology and no abonormal results is a "true" non-failure.
                observations_summary[-1].append(0)
            else:  # everything else is ignored.  It will not be a failure or a starting time at normal.  
                observations_summary[-1].append(-1)
    return observations_summary

# using this one #
def Obs_summary_3(observations): # failure measured from observed 1 to observed 2. Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if np.sum(r[:1,1]) > 0 and np.sum(r[:1,np.array([0,2,3])]) == 0: # if cytology/histolgoy == 1 then that is NOT a failure
                observations_summary[-1].append(0)
            elif np.sum(r[:1,2]) > 0: # if cytology/histolgoy == 2 then that IS a failure 
                observations_summary[-1].append(1)
            else: # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary

# using this one #
def Obs_summary_4(observations): # failure measured from observed cyt/hist 0 to observed cyt/hist ==  2.  Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: # if cytology/histolgoy == 0 and there are no abnormal results, then that is NOT a failure
                observations_summary[-1].append(0)
            elif np.sum(r[:1,2]) > 0: # if cytology/histolgoy == 2 then that IS a failure 
                observations_summary[-1].append(1)
            else: # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary


# using this one #
def Obs_summary_5(observations): # failure measured from observed cyt/hist 0 to observed cyt/hist which is >= 1.  Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: # if cytology/histolgoy == 0 and there are no abnormal results, then that is NOT a failure
                observations_summary[-1].append(0)
            elif np.sum(r[:1,1:]) > 0: # if cytology/histolgoy >= 1 (other than death?) then that IS a failure 
                observations_summary[-1].append(1)
            else: # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary
  
# using this one #
def Obs_summary_6(observations): # failure measured from observed cyt/hist 0 to observed cyt/hist == 3.  Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: # if cytology/histolgoy == 0 and there are no abnormal results, then that is NOT a failure
                observations_summary[-1].append(0)
            elif np.sum(r[:1,3]) > 0: # if cytology/histolgoy == 3 then that IS a failure 
                observations_summary[-1].append(1)
            else: # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary

# using this one #
def Obs_summary_7(observations): # failure measured from observed cyt/hist 0 to observed cyt/hist == 1.  Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: # if cytology/histolgoy == 0 and there are no abnormal results, then that is NOT a failure
                observations_summary[-1].append(0)
            elif np.sum(r[:1,1]) > 0: # if cytology/histolgoy == 1 then that IS a failure 
                observations_summary[-1].append(1)
            else: # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary

# using this one #
def Obs_summary_8(observations): # failure measured from observed 1 to observed 0. Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if np.sum(r[:1,1]) > 0 and np.sum(r[:1,np.array([0,2,3])]) == 0: # if cytology/histolgoy == 1 and there are no other results, then that is NOT a failure
                observations_summary[-1].append(0)
            elif np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: # if cytology/histolgoy == 0 and there are no abnormal results,  then that IS a failure 
                observations_summary[-1].append(1)
            elif np.sum(r[:1,2:]) > 0:
                # keep track of 'cancers'. We will have to discard these patients if there is a cancer between 1 and 0. 
                observations_summary[-1].append(2)
            else: # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary



#######################################

# using this one #
def Obs_summary_9(observations): # failure measured from observed 1 to observed 2. Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if r[0,1] > 0 and np.sum(r[0,np.array([0,2,3])]) == 0 and np.sum(r[1,np.array([2,3])]) == 0: 
            # if cytology == 1 and not 0,2,3 then that is NOT a failure. hist can be zero or one. 
                observations_summary[-1].append(0)
            elif np.sum(r[:1,2]) > 0: 
            # if cytology/histolgoy == 2 then that IS a failure 
                observations_summary[-1].append(1)
            else: 
            # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary

def Obs_summary_10(observations): # failure measured from observed 1 to observed 2. Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if r[0,1] > 0 and np.sum(r[:1,np.array([0,2,3])]) == 0: 
            # if cytology == 1 then that is NOT a failure. hist can't be zero. 
                observations_summary[-1].append(0)
            elif np.sum(r[:1,2]) > 0:
            # if cytology/histolgoy == 2 then that IS a failure 
                observations_summary[-1].append(1)
            else: 
            # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary


# using this one #
def Obs_summary_11(observations): # failure measured from observed 1 to observed 2. Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if r[1,1] > 0 and np.sum(r[:1,np.array([0,2,3])]) == 0: 
            # if histolgoy == 1 then that is NOT a failure. cyt can't be zero.
                observations_summary[-1].append(0)
            elif np.sum(r[:1,2]) > 0: 
            # if cytology/histolgoy == 2 then that IS a failure 
                observations_summary[-1].append(1)
            else: 
            # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary


###########################################

# using this one #
def Obs_summary_12(observations): # failure measured from observed 1 to observed 0. Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if r[0,1] > 0 and np.sum(r[0,np.array([0,2,3])]) == 0 and np.sum(r[1,np.array([2,3])]) == 0:
            # if cytology == 1 then that is NOT a failure. hist can be zero or one. 
                observations_summary[-1].append(0)
            elif np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: 
            # if cytology/histolgoy == 0 and there are no abnormal results,  then that IS a failure 
                observations_summary[-1].append(1)
            else: 
            # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary

def Obs_summary_13(observations): # failure measured from observed 1 to observed 0. Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if r[0,1] > 0 and np.sum(r[:1,np.array([0,2,3])]) == 0: 
            # if cytology == 1 then that is NOT a failure. hist can't be zero. 
                observations_summary[-1].append(0)
            elif np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: 
            # if cytology/histolgoy == 0 and there are no abnormal results,  then that IS a failure 
                observations_summary[-1].append(1)
            else: # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary


# using this one #
def Obs_summary_14(observations): # failure measured from observed 1 to observed 0. Ignore HPV
    observations_summary = []
    for o in observations:
        observations_summary.append([])
        for r in o:
            if r[1,1] > 0 and np.sum(r[:1,np.array([0,2,3])]) == 0: 
            # if histolgoy == 1 then that is NOT a failure. cyt can't be zero.
                observations_summary[-1].append(0)
            elif np.sum(r[:1,0]) > 0 and np.sum(r[:1,1:]) == 0: 
            # if cytology/histolgoy == 0 and there are no abnormal results,  then that IS a failure 
                observations_summary[-1].append(1)
            else: # otherwise it is ignored
                observations_summary[-1].append(-1)
    return observations_summary


#########################################







if __name__ == "__main__":

    
    ### Parallel Library ###
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print 'rank: ', rank
    print 'size: ', size
    
    # this should be in ['inv2','inv3','inv4','inv5','inv8','inv9','inv10','inv11']
    model = sys.argv[1]
    n_patients = 300000
    n_sub_pops = 1 #300
    flag = '1'
    
    do_inv2 = False 
    do_inv3 = False
    do_inv4 = False
    do_inv5 = False
    do_inv6 = False
    do_inv7 = False
    do_inv8 = False
    do_inv9 = False
    do_inv10 = False
    do_inv11 = False
 
    if sys.argv[1] == 'inv2':
        do_inv2 = True
    elif sys.argv[1] == 'inv3':
        do_inv3 = True
    elif sys.argv[1] == 'inv4':
        do_inv4 = True
    elif sys.argv[1] == 'inv5':
        do_inv5 = True
    elif sys.argv[1] == 'inv6':
        do_inv6 = True
    elif sys.argv[1] == 'inv7':
        do_inv7 = True
    elif sys.argv[1] == 'inv8':
        do_inv8 = True
    elif sys.argv[1] == 'inv9':
        do_inv9 = True
    elif sys.argv[1] == 'inv10':
        do_inv10 = True
    elif sys.argv[1] == 'inv11':
        do_inv11 = True

    if do_inv10:
        inv = [16, 23, 25, 30, 35, 40, 45, 50, 60, 200] # have not used. ask Mari and Jan. 
    elif do_inv11: 
        inv = [16, 19, 22, 25, 30, 35, 40, 45, 50, 55, 200 ] # most commonly used so far. 
    elif do_inv9:
        inv = [16,20,25,30,35,40,50,60,200]
    elif do_inv8:
        inv = [16,20,25,30,40,50,60,200]
    elif do_inv5:
        inv = [16,23,30,60, 200]
    elif do_inv4:
        inv = [16,29,69,200] # to try: fewer age bins
    elif do_inv3:
        inv = [16,23,200] 
    elif do_inv2:
        inv  = [16, 200] 
        
        

    if len(sys.argv) > 2:
        age_rule = sys.argv[2] # should be in ['max','dist','none']
    else:
        age_rule = 'none'
    if len(sys.argv) > 3:
        test_rule = sys.argv[3] # should be in ['observed','synthetic']
    else:
        test_rule = 'observed'
    if len(sys.argv) > 4:
        fix_cyt_theta = sys.argv[4] #in ['60','80']
    else:
        fix_cyt_theta = ''






    home = os.path.expanduser('~')   
    
    #input_path = 'Documents/CancerAlgorithms/pmcd/pmcd/lib/GeneralStateModel/Rui' # 'soper3/distributed-data/simulated_markov_chains/%s'%(model)
    input_path = 'soper3/distributed-data/synthetic_data/%s/population%s'%(model,str(rank-1))
    output_path  = 'soper3/distributed-data/kaplan_meier_curves/%s'%(model)

    
    #data_location = '../distributed_updated_nonzero_data/'
    data_location = 'soper3/distributed-data/distributed_updated_data/'
    #data_location = 'Documents/CancerAlgorithms/pmcd/pmcd/lib/GeneralStateModel/distributed_updated_data/'
    #data_location = '../survey_patient_data/'

    do_truncate_ages = False
    min_age = 16
    nStates = nDiseaseStates = 4
    nTests = 3
    
    if test_rule == 'observed':
        use_observed_tests = True
    else:
        use_observed_tests = False 
     
    if age_rule == 'max':
        do_month_max = True
        do_month_dist = False
    elif age_rule == 'dist':
        do_month_max = False
        do_month_dist = True
    else:
        do_month_max = False
        do_month_dist = False
    
    

    output_flag = '{}_{}'.format(age_rule, test_rule)

    if len(fix_cyt_theta) > 0:
        output_flag += '_cyt{}'.format(fix_cyt_theta)



    observation_patterns =  [3,4,7,8]      #   [3,8,9,10,11,12,13,14] # [2,4,5,6,7]
    
    #referrence_age_ranges = [ (16,19), (19,22), (22,25), (25,30), (30,35), (35,40), (40,45), (45,50), (50,55), (55,200), (23,200), (16,29), (30,69), (70,200) ]
    #referrence_age_ranges = [ (16,20), (20,25), (25,30), (30,35), (35,40), (40,45), (45,50), (50,55), (55,200), (23,200) ]
    referrence_age_ranges = [ (16,20), (20,25), (25,30), (30,35), (35,40), (40,50), (50,60), (60,200), (25,200), (16,30), (30,60), (60,200) ]
    
    times           =   []
    observations    =   []
    regressors      =   []
    ages = []

     
    
    print 'loading data on proc ', rank
    if rank == 0:
        # load real data (not synthetic)
        temp_ages = []
        for r in range(n_patients/100):   
      
            #            
            times_path  = os.path.join(home,data_location,'p{}'.format(r),'mcmcPatientTimes')
            test_path   = os.path.join(home,data_location,'p{}'.format(r),'mcmcPatients')
            obs_path    = os.path.join(home,data_location,'p{}'.format(r),'mcmcPatientObservations')
            reg_path    = os.path.join(home,data_location,'p{}'.format(r),'mcmcPatientRegressors')
        
            times         +=  pickle.load(open(times_path , 'r'))
            observations  +=  pickle.load(open(obs_path   , 'r'))
            regressors     =  pickle.load(open(reg_path   , 'r'))  
            temp_ages    +=  regressors[1] 
     
    
        # Reset age
        for temp_patient_ages, patient_times in zip(temp_ages, times):
            new_patient_ages = temp_patient_ages[0] + patient_times/12.0
            ages.append(new_patient_ages)
       
    else:
        for r in range(n_sub_pops):  
            
            # these are the new observation file names. Some combos of flag do not exist. need to use the old. 
            obs_path   = 'simulated_observations_population{}_proc{}_{}_{}'.format(rank-1,r,model, output_flag)
            
            # these are the original observation file names. 
            obs_path_alt   = 'simulated_observations_flag{}_population{}_proc{}_{}'.format(flag,rank-1,r,model)
            
            # these should be the same for all. 
            times_path = 'simulated_times_population{}_proc{}_{}'.format(rank-1,r,model)
            ages_path  = 'simulated_ages_population{}_proc{}_{}'.format(rank-1,r,model)
            
            if os.path.isfile(os.path.join(home,input_path,obs_path  )):
                observations  += pickle.load(open(os.path.join(home,input_path,obs_path  ),'r'))
            else:
                print 'WARNING: Using old observed data files.  Newer observed data does not exist.'
                observations  += pickle.load(open(os.path.join(home,input_path,obs_path_alt  ),'r'))

            times         += pickle.load(open(os.path.join(home,input_path,times_path),'r'))
            ages          += pickle.load(open(os.path.join(home,input_path,ages_path ),'r'))
            


    
    print 'computing KP estimator on proc ', rank
    for referrence_age_range in referrence_age_ranges:
        print 'reference age range {}-{} on proc {}'.format(referrence_age_range[0], referrence_age_range[1],rank)
        for obs in observation_patterns:
            print 'observations pattern {} on proc {}'.format(obs,rank)
            # Compute Kaplan-Meier-like Estimator #
            observations_summary, which_obs = Obs_summary(observations,obs)  
            reduced_inter_event_times, KP_vector = KP_stat(referrence_age_range, ages, times, observations_summary)
                     
            KP_plot = [[k,k] for k in KP_vector]
            KP_plot = [k for K in KP_plot for k in K]
            t_plot = [[t,t] for t in np.sort([0] + reduced_inter_event_times)]
            t_plot = [t for T in t_plot for t in T ]
                

            print 'saving results on proc ', rank
            if rank == 0:
                
                file_name = 'KM_estimator_real_data_{}_{}_{}-{}'.format(output_flag,
                                                                     str(obs),
                                                                     str(referrence_age_range[0]),
                                                                     str(referrence_age_range[1]))
                f = open(os.path.join(home, output_path , file_name), 'w')
                pickle.dump(KP_plot,f)
                f.close()

                
                file_name = 'KM_times_real_data_{}_{}_{}-{}'.format(output_flag,
                                                                 str(obs),
                                                                 str(referrence_age_range[0]),
                                                                 str(referrence_age_range[1]))
                f = open(os.path.join(home, output_path , file_name), 'w')
                pickle.dump(t_plot,f)
                f.close()      
                
            else: 
                
                file_name = 'KM_estimator_population{}_{}_{}_{}-{}'.format(str(rank-1),
                                                                        output_flag,
                                                                        str(obs),
                                                                        str(referrence_age_range[0]),
                                                                        str(referrence_age_range[1]))
                f = open(os.path.join(home, output_path , file_name), 'w')
                pickle.dump(KP_plot,f)
                f.close()

                
                file_name = 'KM_times_population{}_{}_{}_{}-{}'.format(str(rank-1),
                                                                    output_flag,
                                                                    str(obs),
                                                                    str(referrence_age_range[0]),
                                                                    str(referrence_age_range[1]))
                f = open(os.path.join(home, output_path , file_name), 'w')
                pickle.dump(t_plot,f)
                f.close() 
            
            
            ### This is not necessary ###
            # I can print it later #
            '''
            KP_plot_list  = []
            t_plot_list = []
            
            KP_plot_list = comm.gather(KP_plot,root=0)
            t_plot_list = comm.gather(t_plot,root=0)
            
            ### in order to do this we must gather all the KP_plots from the other procs...
            if rank == 0:
                plt.figure(str(referrence_age_range[0]) + '-' + str(referrence_age_range[1]) )
                plt.plot(np.array(t_plot_list[0][1:])/12., KP_plot_list[0][:-1],label='observed data',color='r')
                plt.ylim([0.8,1])
            
            
                plt.plot(np.array(t_plot_list[1][1:])/12., KP_plot_list[1][:-1],color='b',alpha=0.5)#,label='simulated data')
                
                for r in range(1,size):
                    plt.plot(np.array(t_plot_list[r+1][1:])/12., KP_plot_list[r+1][:-1],color='b',alpha=0.5)#,label='simulated data')
                plt.legend()
            
            
                file_name = 'KP_obs%s_%s_%s'%(str(obs),str(referrence_age_range[0]),str(referrence_age_range[1]))
                plt.savefig(os.path.join(home, output_path , file_name))
        
            '''    
        
    

