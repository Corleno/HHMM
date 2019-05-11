1. Experiments: 

sbatch
dir: src/Experiments/
5 different sbatch scripts: "bootstrapt_sbatch_i", where i = 0,1,2,3,4.


2. Source codes: 

inference
dir: 
src/EM_exact_death_times_hierarchical_shared.py

This code has following arguments:
name: name of the experiment
min_age: minimum age, default = 16
dataset: data specification, default = updated_data
age_inv: age interval specification, default = inv4, which is [16,23), [23,30), [30, 60), [60, inf)
n_patients_per_proc: number of patients per process, default = 100.
max_steps_em: number of EM iterations, default = 100
max_steps_optim: maximum number of optimal iterations in the M step, default = 5.
model: discrete (Markov Chain) and continuous (Markov Process), default = continuous.
test: boolean indicator for testing, default = store_true
autograd_optim: boolean indicator for auto-gradient in optimization (M-step).
Z_prior: prior probability of Model 1 (prior proportion of potential patients).
bootstrap_seed: seed number setting in Boostrap, default = 0.
bootstrap_total: number of total seeds, defualt = 5000

validation
dir: src/model_validation/model_validation.py

name: name of the experiment, default = Model_Validation
min_age: minimum age default = 16
dataset: data specification, default = updated_data
age_inv: age interval specification, default = inv4, which is [16,23), [23,30), [30, 60), [60, inf)
n_patients_per_proc: number of patients per process, default = 100.
test: boolean indicator for testing, default = store_true

data_process
dir: src/combine_distributed_data.py

It combines all/part of distributed data (distributed_updated_nonzero_data, distributed_updated_data) to a integral data saved in (data_full (14601 distributed file and each file include 100 patient data), data_1000, data_nonzero_1000)
The save structure has following features:
1. mcmcPatientTestTypes: testTypes[p] patient_tests, patient_tests[j] patient_test ([3,], 3 categories) 
2. mcmcPatientObservations: observations[p] patient_observations, patient_observations[j] patient_observation ([3,4], count)
3. mcmcPatientAges: ages[p] patient_ages, patient_ages[j] patient_age (scaler)

4. mcmcPatientTreatmentIndx: treatment_indx[p] patient_treatment_indx (array)
5. mcmcPatientCensorDates: censor_ages[p] censor_age (scalar)
6. mcmcPatientDeathStates: death_states[p] death_state ([0,1])


3. Data:

dir: updated_data -> "data/distributed_updated_data/"

4. Results:
dir: 
### original res/ 

