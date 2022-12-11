# HHMM
Hierarchical Hidden Markov Model

# Code 

1. Experiment batch script:
   - Batch job script for model training: `Experiments/EM_bootstrapt_sbatch` 
2. Clinical applications:
   - Codes in clinical_applications is used for risk stratification. Basically, 
   it compares our HHMM based method with baseline method for risk stratification in
   terms of KM curve metric.
3. Model comparison:
   - This compare our model with naive HMM model as well as a batch of deep learning 
   based models in terms of model prediction performance.
   - In particular, 
     - Code for HHMM is in `Model_comparison/EM_hierechical_prediction.py`.
     - Code for HMM is in `Model_comparison/EM_prediction.py`.
     - Codes for deep learning methods are in `Model_comparison/deep_learning`.
4. Model inference:
   - We have two versions of inference code. 
     - One for python2 in `Model_inference/python2/HHMM_inference.py`. 
     - The other for python3 in `Model_inference/python3/EM_hierarchical_paralleled.py` with the 
     bootstrap version in `Model_inference/python3/EM_hierarchical_paralleled_bootstrap.py`.
5. Model prediction:
   - Codes in this folder are used to predict the score for three screening results based on historical records.
   - In particular,
     - Prediction code for HHMM is in `Model_prediction/EM_hierechical_prediction_SR.py`.
     - Prediction code for HMM is in `Model_prediction/EM_prediction_SR.py`.
6. Model validation:
   - Code to plot Kaplan Meier curve given HMM and HHMM models is provided in `Model_validation/model_validation.py`.
7. Simulation:
   - To illustrate the correctness of inference code, we conduct synthetic experiments in which date are generated from 
   HHMM. All simulation code and corresponding inference codes are provided in `Simulation`.
8. Survey_applications:
   - Codes for HHMM model based application with survey data are provided in `Survey_applications`.
9. Others:
   - We provided the data analysis codes and ultility codes in `Others`.

# Data
Data are available for inquiry.
