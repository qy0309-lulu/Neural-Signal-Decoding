# SET PARAMETER
save_folder='/root/KF_Results/'

load_folder='/root/'


# Dataset you're using

# dataset='s1'
# dataset='m1'
dataset='hc'


# ## 1. Import Packages

#Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
import sys
import bayes_opt

#Add the main folder to the path, so we have access to the files there.
#Note that if your working directory is not the Paper_code folder, you may need to manually specify the path to the main folder. For example: sys.path.append('/home/jglaser/GitProj/Neural_Decoding')
sys.path.append('../..')

#Import function to get the covariate matrix that includes spike history from previous bins
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

#Import metrics
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho

#Import decoder functions
from Neural_Decoding.decoders import KalmanFilterDecoderGPU

from bayes_opt import BayesianOptimization

# 屏蔽 XLA 编译器日志
import tensorflow as tf


#Turn off deprecation warnings

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ## 2. Load Data

if dataset=='s1':
    with open(load_folder+'example_data_s1.pickle','rb') as f:
    #     neural_data,vels_binned=pickle.load(f,encoding='latin1')
        neural_data,vels_binned=pickle.load(f)

if dataset=='m1':
    with open(load_folder+'example_data_m1.pickle','rb') as f:
    #     neural_data,vels_binned=pickle.load(f,encoding='latin1')
        neural_data,vels_binned=pickle.load(f)

if dataset=='hc':
    with open(load_folder+'example_data_hc.pickle','rb') as f:
    #     neural_data,pos_binned=pickle.load(f,encoding='latin1')
        neural_data,pos_binned=pickle.load(f)


# ## 3. Preprocess Data

# ### 3A. Format Covariates

# #### Format Input Covariates

#Remove neurons with too few spikes in HC dataset
if dataset=='hc':
    nd_sum=np.nansum(neural_data,axis=0)
    rmv_nrn=np.where(nd_sum<100)
    neural_data=np.delete(neural_data,rmv_nrn,1)

#The covariate is simply the matrix of firing rates for all neurons over time
X_kf=neural_data


# #### Format Output Covariates


if dataset=='s1' or dataset=='m1':

    #We will now determine position
    pos_binned=np.zeros(vels_binned.shape) #Initialize 
    pos_binned[0,:]=0 #Assume starting position is at [0,0]
    #Loop through time bins and determine positions based on the velocities
    for i in range(pos_binned.shape[0]-1): 
        pos_binned[i+1,0]=pos_binned[i,0]+vels_binned[i,0]*.05 #Note that .05 is the length of the time bin
        pos_binned[i+1,1]=pos_binned[i,1]+vels_binned[i,1]*.05

    #We will now determine acceleration    
    temp=np.diff(vels_binned,axis=0) #The acceleration is the difference in velocities across time bins 
    acc_binned=np.concatenate((temp,temp[-1:,:]),axis=0) #Assume acceleration at last time point is same as 2nd to last

    #The final output covariates include position, velocity, and acceleration
    y_kf=np.concatenate((pos_binned,vels_binned,acc_binned),axis=1)


if dataset=='hc':

    temp=np.diff(pos_binned,axis=0) #Velocity is the difference in positions across time bins
    vels_binned=np.concatenate((temp,temp[-1:,:]),axis=0) #Assume velocity at last time point is same as 2nd to last

    temp2=np.diff(vels_binned,axis=0) #The acceleration is the difference in velocities across time bins 
    acc_binned=np.concatenate((temp2,temp2[-1:,:]),axis=0) #Assume acceleration at last time point is same as 2nd to last

    #The final output covariates include position, velocity, and acceleration
    y_kf=np.concatenate((pos_binned,vels_binned,acc_binned),axis=1)


# #### In HC dataset, remove time bins with no output (y value)

if dataset=='hc':
    rmv_time=np.where(np.isnan(y_kf[:,0]) | np.isnan(y_kf[:,1]))
    X_kf=np.delete(X_kf,rmv_time,0)
    y_kf=np.delete(y_kf,rmv_time,0)


# ** In HC dataset, there is a long period without movement starting at ~80%, so we only use the first 80% of the data**

if dataset=='hc':
    X_kf=X_kf[:int(.8*X_kf.shape[0]),:]
    y_kf=y_kf[:int(.8*y_kf.shape[0]),:]


# ### 3B. Define training/testing/validation sets
# We have 10 cross-validation folds. In each fold, 10% of the data is a test set, 10% is a validation set, and 80% is the training set. So in the first fold, for example, 0-10% is validation, 10-20% is testing, and 20-100% is training.

valid_range_all=[[0,.1],[.1,.2],[.2,.3],[.3,.4],[.4,.5],
                 [.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1]]
testing_range_all=[[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],
                 [.6,.7],[.7,.8],[.8,.9],[.9,1],[0,.1]]
#Note that the training set is not aways contiguous. For example, in the second fold, the training set has 0-10% and 30-100%.
#In that example, we enter of list of lists: [[0,.1],[.3,1]]
training_range_all=[[[.2,1]],[[0,.1],[.3,1]],[[0,.2],[.4,1]],[[0,.3],[.5,1]],[[0,.4],[.6,1]],
                   [[0,.5],[.7,1]],[[0,.6],[.8,1]],[[0,.7],[.9,1]],[[0,.8]],[[.1,.9]]]

num_folds=len(valid_range_all) #Number of cross validation folds


# ## 4. Run CV

# **Initialize lists of results**

#R2 values
mean_r2_kf=np.empty(num_folds)

#Actual data
y_kf_test_all=[]
y_kf_train_all=[]
y_kf_valid_all=[]

#Test/training/validation predictions
y_pred_kf_all=[] 
y_train_pred_kf_all=[]
y_valid_pred_kf_all=[]


# **In the following section, we**
# 1. Loop across folds
# 2. Extract the training/validation/testing data
# 3. Preprocess the data
# 4. Run the KF decoder (including the hyperparameter optimization)
# 5. Save the results

num_examples=X_kf.shape[0] #number of examples (rows in the X matrix)

for i in range(num_folds): #Loop through the folds

    ######### SPLIT DATA INTO TRAINING/TESTING/VALIDATION #########

    #Note that all sets have a buffer of 1 bin at the beginning and 1 bin at the end 
    #This makes it so that the different sets don't include overlapping neural data

    #This differs from having buffers of "num_bins_before" and "num_bins_after" in the other datasets, 
    #which creates a slight offset in time indexes between these results and those from the other decoders

    #Get testing set for this fold
    testing_range=testing_range_all[i]
    testing_set=np.arange(int(np.round(testing_range[0]*num_examples))+1,int(np.round(testing_range[1]*num_examples))-1)

    #Get validation set for this fold
    valid_range=valid_range_all[i]
    valid_set=np.arange(int(np.round(valid_range[0]*num_examples))+1,int(np.round(valid_range[1]*num_examples))-1)

    #Get training set for this fold
    #Note this needs to take into account a non-contiguous training set (see section 3B)
    training_ranges=training_range_all[i]
    for j in range(len(training_ranges)): #Go through different separated portions of the training set
        training_range=training_ranges[j]
        if j==0: #If it's the first portion of the training set, make it the training set
            training_set=np.arange(int(np.round(training_range[0]*num_examples))+1,int(np.round(training_range[1]*num_examples))-1)
        if j==1: #If it's the second portion of the training set, concatentate it to the first
            training_set_temp=np.arange(int(np.round(training_range[0]*num_examples))+1,int(np.round(training_range[1]*num_examples))-1)
            training_set=np.concatenate((training_set,training_set_temp),axis=0)

    #Get training data
    X_kf_train=X_kf[training_set,:]
    y_kf_train=y_kf[training_set,:]

    #Get validation data
    X_kf_valid=X_kf[valid_set,:]
    y_kf_valid=y_kf[valid_set,:]

    #Get testing data
    X_kf_test=X_kf[testing_set,:]
    y_kf_test=y_kf[testing_set,:]


    ##### PREPROCESS DATA #####

    #Z-score "X_kf" inputs. 
    X_kf_train_mean=np.nanmean(X_kf_train,axis=0) #Mean of training data
    X_kf_train_std=np.nanstd(X_kf_train,axis=0) #Stdev of training data
    X_kf_train=(X_kf_train-X_kf_train_mean)/X_kf_train_std #Z-score training data
    X_kf_test=(X_kf_test-X_kf_train_mean)/X_kf_train_std #Preprocess testing data in same manner as training data
    X_kf_valid=(X_kf_valid-X_kf_train_mean)/X_kf_train_std #Preprocess validation data in same manner as training data

    #Zero-center outputs
    y_kf_train_mean=np.nanmean(y_kf_train,axis=0) #Mean of training data outputs
    y_kf_train=y_kf_train-y_kf_train_mean #Zero-center training output
    y_kf_test=y_kf_test-y_kf_train_mean #Preprocess testing data in same manner as training data
    y_kf_valid=y_kf_valid-y_kf_train_mean #Preprocess validation data in same manner as training data  


    ####### RUN KALMAN FILTER #######

    #We are going to loop through different lags, and for each lag: 
        #-we will find the optimal hyperparameter C based on the validation set R2
        #-with that hyperparameter, we will get the validation set R2 for the given lag
    #We will determine the lag as the one that gives the best validation set R2
    #Finally, using the lag and hyperparameters determined (based on above), we will get the test set R2


    #First, we set the limits of lags that we will evaluate for each dataset
    if dataset=='hc':
        valid_lags=np.arange(-5,6)
    if dataset=='m1':
        valid_lags=np.arange(-10,1)
    if dataset=='s1':
        valid_lags=np.arange(-6,7)
    num_valid_lags=valid_lags.shape[0] #Number of lags we will consider

    #Initializations
    lag_results=np.empty(num_valid_lags) #Array to store validation R2 results for each lag
    C_results=np.empty(num_valid_lags) #Array to store the best hyperparameter for each lag




    #### Wrapper function that returns the best validation set R2 for each lag
    #That is, for the given lag, it will find the best hyperparameters to maximize validation set R2
    #and the function returns that R2 value
    def kf_evaluate_lag(lag,X_kf_train,y_kf_train,X_kf_valid,y_kf_valid):    

        #Re-align data to take lag into account
        if lag<0:
            y_kf_train=y_kf_train[-lag:,:]
            X_kf_train=X_kf_train[:lag,:]
            y_kf_valid=y_kf_valid[-lag:,:]
            X_kf_valid=X_kf_valid[:lag,:]
        if lag>0:
            y_kf_train=y_kf_train[0:-lag,:]
            X_kf_train=X_kf_train[lag:,:]
            y_kf_valid=y_kf_valid[0:-lag,:]
            X_kf_valid=X_kf_valid[lag:,:]

        #This is a function that evaluates the Kalman filter for the given hyperparameter C
        #and returns the R2 value for the hyperparameter. It's used within Bayesian optimization
        def kf_evaluate(C):
            model_kf=KalmanFilterDecoderGPU(C=C) #Define model
            model_kf.fit(X_kf_train,y_kf_train) #Fit model
            y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid) #Get validation set predictions
            #Get validation set R2
            if dataset=='hc':
                return np.mean(get_R2(y_kf_valid,y_valid_predicted_kf)[0:2]) #Position is components 0 and 1
            if dataset=='m1' or dataset=='s1':
                return np.mean(get_R2(y_kf_valid,y_valid_predicted_kf)[2:4]) #Velocity is components 2 and 3

        #Do Bayesian optimization!
        kfBO = BayesianOptimization(kf_evaluate, {'C': (.5, 20)}, verbose=0) #Define Bayesian optimization, and set limits of hyperparameters
        kfBO.maximize(init_points=10, n_iter=10) #Set number of initial runs and subsequent tests, and do the optimization
        best_params=kfBO.max['params'] #Get the hyperparameters that give rise to the best fit
        C=best_params['C']
#         print("C=", C)

        #Get the validation set R2 using the best hyperparameters fit above:    
        model_kf=KalmanFilterDecoderGPU(C=C) #Define model
        model_kf.fit(X_kf_train,y_kf_train) #Fit model
        y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid) #Get validation set predictions
        #Get validation set R2
        if dataset=='hc':
            return [np.mean(get_R2(y_kf_valid,y_valid_predicted_kf)[0:2]), C] #Position is components 0 and 1
        if dataset=='m1' or dataset=='s1':
            return [np.mean(get_R2(y_kf_valid,y_valid_predicted_kf)[2:4]), C] #Velocity is components 2 and 3


    ### Loop through lags and get validation set R2 for each lag ####

    for j in range(num_valid_lags):    
        valid_lag=valid_lags[j] #Set what lag you're using
        #Run the wrapper function, and put the R2 value and corresponding C (hyperparameter) in arrays
        [lag_results[j],C_results[j]]=kf_evaluate_lag(valid_lag,X_kf_train,y_kf_train,X_kf_valid,y_kf_valid)



    #### Get results on test set ####

    #Get the lag (and corresponding C value) that gave the best validation results
    lag=valid_lags[np.argmax(lag_results)] #The lag
#     print("lag=",lag)
    C=C_results[np.argmax(lag_results)] #The hyperparameter C    

    #Re-align data to take lag into account
    if lag<0:
        y_kf_train=y_kf_train[-lag:,:]
        X_kf_train=X_kf_train[:lag,:]
        y_kf_test=y_kf_test[-lag:,:]
        X_kf_test=X_kf_test[:lag,:]
        y_kf_valid=y_kf_valid[-lag:,:]
        X_kf_valid=X_kf_valid[:lag,:]
    if lag>0:
        y_kf_train=y_kf_train[0:-lag,:]
        X_kf_train=X_kf_train[lag:,:]
        y_kf_test=y_kf_test[0:-lag,:]
        X_kf_test=X_kf_test[lag:,:]
        y_kf_valid=y_kf_valid[0:-lag,:]
        X_kf_valid=X_kf_valid[lag:,:]

    #Run the Kalman filter
    model_kf=KalmanFilterDecoderGPU(C=C) #Define model
    print(f"\n start training fold{i}:")
    model_kf.fit(X_kf_train,y_kf_train) #Fit model
    y_test_predicted_kf=model_kf.predict(X_kf_test,y_kf_test) #Get test set predictions
    #Get test set R2 values and put them in arrays
    if dataset=='hc':
        mean_r2_kf[i]=np.mean(get_R2(y_kf_test,y_test_predicted_kf)[0:2]) #Position is components 0 and 1
        print(f"fold{i}_R2:",np.mean(get_R2(y_kf_test,y_test_predicted_kf)[0:2]))
    if dataset=='m1' or dataset=='s1':
        mean_r2_kf[i]=np.mean(get_R2(y_kf_test,y_test_predicted_kf)[2:4]) #Velocity is components 2 and 3
        print(f"fold{i}_R2:",np.mean(get_R2(y_kf_test,y_test_predicted_kf)[2:4]))



    ### Add variables to list (for saving) ###
    y_kf_test_all.append(y_kf_test)
    y_kf_valid_all.append(y_kf_valid)    
    y_kf_train_all.append(y_kf_train)    

    y_pred_kf_all.append(y_test_predicted_kf)
    y_valid_pred_kf_all.append(model_kf.predict(X_kf_valid,y_kf_valid))
    y_train_pred_kf_all.append(model_kf.predict(X_kf_train,y_kf_train))    


    ### Save ###
    # with open(save_folder+dataset+'_results_kf2.pickle','wb') as f:
    #     pickle.dump([mean_r2_kf,y_pred_kf_all,y_valid_pred_kf_all,y_train_pred_kf_all,
    #                  y_kf_test_all,y_kf_valid_all,y_kf_train_all],f)    


# print(f"y_kf_test_all shape:{y_kf_test_all}")
# print(f"y_pred_kf_all shape:{y_pred_kf_all}")
plt.figure(figsize=(12,4))
plt.plot(y_kf_test_all[1][0:1000,0],'b',label='test')
plt.plot(y_pred_kf_all[1][0:1000,0],'r',label='train')
plt.legend()
plt.savefig(r'KF_FullData.jpg',dpi=300)








