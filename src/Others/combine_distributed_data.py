import pickle

# data_location = 'distributed_updated_nonzero_data/'
data_location = 'distributed_updated_data/'
ID_0 = []
testTypes_0 = []
observations_0 = []
treatment_indx_0 = []
ages_0 = []
censor_ages_0 = []
death_states_0 = []

# for i in range(14601):
for i in range(1000):
    print("Combined the {}th folder".format(i))
    subdata_location = data_location + 'p%s/'%(str(i))
    times           =  pickle.load(open(subdata_location + 'mcmcPatientTimes', 'rb'), encoding = "bytes")
    regressors      =  pickle.load(open(subdata_location + 'mcmcPatientRegressors', 'rb'), encoding = "bytes")
    testTypes       =  pickle.load(open(subdata_location + 'mcmcPatientTestTypes', 'rb'), encoding = "bytes")
    observations    =  pickle.load(open(subdata_location + 'mcmcPatientObservations', 'rb'), encoding = "bytes")
    treatment_indx  =  pickle.load(open(subdata_location + 'mcmcPatientTreatmentIndx', 'rb'), encoding = "bytes")
    censor_ages     =  pickle.load(open(subdata_location + 'mcmcPatientCensorDates', 'rb'), encoding = "bytes")             #
    death_states    =  pickle.load(open(subdata_location + 'mcmcPatientDeathStates', 'rb'), encoding = "bytes")       
    temp_ages = regressors[1]
    ages = []
    # Reset age
    for temp_patient_ages, patient_times in zip(temp_ages, times):
        new_patient_ages = temp_patient_ages[0] + patient_times/12.0
        ages.append(new_patient_ages)
    testTypes_0 += testTypes
    observations_0 += observations
    treatment_indx_0 += treatment_indx
    ages_0 += ages
    censor_ages_0 += censor_ages
    death_states_0 += death_states

# Rename
ID = ID_0
testTypes = testTypes_0
observations = observations_0
treatment_indx = treatment_indx_0
ages = ages_0
censor_ages = censor_ages_0
death_states = death_states_0

# save_location = 'data_full/'
save_location = 'data_1000/'
# save_location = 'data_nonzero_1000/'
with open(save_location + 'mcmcPatientTestTypes', 'wb') as res:
    pickle.dump(testTypes, res)
with open(save_location + 'mcmcPatientObservations', 'wb') as res:
    pickle.dump(observations, res)
with open(save_location + 'mcmcPatientTreatmentIndx', 'wb') as res:
    pickle.dump(treatment_indx, res)
with open(save_location + 'mcmcPatientAges', 'wb') as res:
    pickle.dump(ages, res)
with open(save_location + 'mcmcPatientCensorDates', 'wb') as res:
    pickle.dump(censor_ages, res)
with open(save_location + 'mcmcPatientDeathStates', 'wb') as res:
    pickle.dump(death_states, res)