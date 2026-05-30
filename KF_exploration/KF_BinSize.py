#!/usr/bin/env python
# coding: utf-8

# # Kalman Filter with changing bin sizes

# ## User Options

save_folder='/home/jglaser/Files/Neural_Decoding/Results/'


# Folder you're loading from

# load_folder=''
load_folder='/home/jglaser/Data/DecData/'


# Dataset you're using

# dataset='s1'
# dataset='m1'
dataset='hc'


# ## 1. Import Packages
# 
# We import both standard packages, and functions from the accompanying .py files

#Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
import sys
import time

#Add the main folder to the path, so we have access to the files there.
#Note that if your working directory is not the Paper_code folder, you may need to manually specify the path to the main folder. For example: sys.path.append('/home/jglaser/GitProj/Neural_Decoding')
sys.path.append('../..')

#Import function to get the covariate matrix that includes spike history from previous bins
from preprocessing_funcs import get_spikes_with_history

###Import functions for binning data for preprocessing###
from preprocessing_funcs import bin_spikes
from preprocessing_funcs import bin_output

#Import metrics
from metrics import get_R2
from metrics import get_rho

#Import decoder functions
from decoders import KalmanFilterDecoder

from bayes_opt import BayesianOptimization


# In[5]:


#Turn off deprecation warnings

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ## 2. Load Data
# 
# Here, we load data in the more raw format, since the preprocessing depends on the bin size (which we will vary later on)

# In[6]:


if dataset=='s1':    
    data=io.loadmat(load_folder+'s1_data_raw.mat')    
    spike_times=data['spike_times'] #Load spike times of all neurons
    vels=data['vels'] #Load x and y velocities
    vel_times=data['vel_times'] #Load times at which velocities were recorded

if dataset=='m1':    
    data=io.loadmat(load_folder+'m1_data_raw.mat')    
    spike_times=data['spike_times'] #Load spike times of all neurons
    vels=data['vels'] #Load x and y velocities
    vel_times=data['vel_times'] #Load times at which velocities were recorded    

if dataset=='hc':
    data=io.loadmat(load_folder+'hc_data_raw.mat')
    spike_times=data['spike_times'] #Load spike times of all neurons
    pos=data['pos'] #Load x and y positions
    pos_times=data['pos_times'][0] #Load times at which positions were recorded      


# ## 3. Preprocess Data

# Options for extracting the data (these are copied from the preprocessing file)

# In[7]:


## USER OPTIONS
if dataset=='s1' or dataset=='m1':
    t_start=vel_times[0] #Time to start extracting data - here the first time velocity was recorded
    t_end=vel_times[-1] #Time to finish extracting data - here the last time velocity was recorded

if dataset=='hc':    
    t_start=pos_times[0] #Time to start extracting data - here the first time position was recorded
    t_end=5608

downsample_factor=1 #Downsampling of output (to make binning go faster). 1 means no downsampling.


#When loading the Matlab cell "spike_times", Python puts it in a format with an extra unnecessary dimension
#First, we will put spike_times in a cleaner format: an array of arrays
spike_times=np.squeeze(spike_times)
for i in range(spike_times.shape[0]):
    spike_times[i]=np.squeeze(spike_times[i])


# Set the bin sizes we will test

# In[8]:


if dataset=='s1' or dataset=='m1':
    dts=[.01,.02,.03,.04,.05,.1] #Size of time bins (in seconds)

if dataset=='hc':
    dts=[.03,.05,.1,.2,.4] #Size of time bins (in seconds)

num_folds=len(dts) #Number of loops we'll do (I'm just calling it "folds" so I can keep old code that used CV folds)


# Initializations of lists/vectors

# In[9]:


mean_r2_kf=np.empty(num_folds)

y_kf_test_all=[]
y_kf_train_all=[]
y_kf_valid_all=[]

y_pred_kf_all=[] #Test predictions
y_train_pred_kf_all=[]
y_valid_pred_kf_all=[]


# ## Loop over bin sizes and do everything!

# In[10]:


for i in range(num_folds): #Loop over bin sizes
    dt=dts[i] #Get the bin size for the current loop


    #### FORMAT OUTPUT ####

    #Bin output (velocity) data using "bin_output" function
    if dataset=='s1' or dataset=='m1':
        vels_binned=bin_output(vels,vel_times,dt,t_start,t_end,downsample_factor)
    if dataset=='hc':
        pos_binned=bin_output(pos,pos_times,dt,t_start,t_end,downsample_factor)

    #For the Kalman filter, we use the position, velocity, and acceleration as outputs
    #Ultimately, we are only concerned with the goodness of fit of velocity (s1 or m1) or position (hc)
    #But using them all as covariates helps performance

    if dataset=='s1' or dataset=='m1':

        #We will now determine position
        pos_binned=np.zeros(vels_binned.shape) #Initialize 
        pos_binned[0,:]=0 #Assume starting position is at [0,0]
        #Loop through time bins and determine positions based on the velocities
        for k in range(pos_binned.shape[0]-1): 
            pos_binned[k+1,0]=pos_binned[k,0]+vels_binned[k,0]*dt
            pos_binned[k+1,1]=pos_binned[k,1]+vels_binned[k,1]*dt

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


    #### FORMAT INPUT ####

    #Bin neural data using "bin_spikes" function
    neural_data=bin_spikes(spike_times,dt,t_start,t_end)

    #Remove neurons with too few spikes in HC dataset
    if dataset=='hc':
        nd_sum=np.nansum(neural_data,axis=0)
        rmv_nrn=np.where(nd_sum<100)
        neural_data=np.delete(neural_data,rmv_nrn,1)


    #The covariate is simply the matrix of firing rates for all neurons over time
    X_kf=neural_data


    # In HC dataset, remove time bins with no output (y value)
    if dataset=='hc':
        #Remove time bins with no output (y value)
        rmv_time=np.where(np.isnan(y_kf[:,0]) | np.isnan(y_kf[:,1]))
        X_kf=np.delete(X_kf,rmv_time,0)
        y_kf=np.delete(y_kf,rmv_time,0)



    #### Define training/testing/validation sets ######

    if dataset=='hc':

        test_size=int(450/dt) #7.5 min
        end_idx=np.int(X_kf.shape[0]*.8)
        tr_end_idx=end_idx-2*test_size
        train_size=int(2250/dt)

#         testing_range=[end_idx-2*test_size,end_idx-test_size]
#         valid_range=[end_idx-test_size,end_idx]

        valid_range=[end_idx-2*test_size,end_idx-test_size]
        testing_range=[end_idx-test_size,end_idx]
        training_range=[tr_end_idx-train_size,tr_end_idx]

    if dataset=='s1' or dataset=='m1':

        if dataset=='s1':
            test_size=int(300/dt) #5 min
            end_idx=np.int(X_kf.shape[0]*.9)
            train_size=int(1200/dt) # 20 min
        if dataset=='m1':
            test_size=int(300/dt) #5 min
            end_idx=np.int(X_kf.shape[0]*1)
            train_size=int(600/dt) # 10 min

        tr_end_idx=end_idx-2*test_size

        valid_range=[end_idx-2*test_size,end_idx-test_size]
        testing_range=[end_idx-test_size,end_idx]
        training_range=[tr_end_idx-train_size,tr_end_idx]   


    ###### RUN DECODERS #######

    t1=time.time()

    num_examples=X_kf.shape[0]


    ######### SPLIT DATA INTO TRAINING/TESTING/VALIDATION #########


    #Note that all sets have a buffer of 1 bins at the beginning, and 1 bins at the end
    #This makes it so that the different sets don't include overlapping neural data

    #Testing set
    testing_set=np.arange(testing_range[0]+1,testing_range[1]-1)


    #Validation set
    valid_set=np.arange(valid_range[0]+1,valid_range[1]-1)

    #Training_set
    training_set=np.arange(training_range[0]+1,training_range[1]-1)


    #Get training data
    X_kf_train=X_kf[training_set,:]
    y_kf_train=y_kf[training_set,:]

    #Get validation data
    X_kf_valid=X_kf[valid_set,:]
    y_kf_valid=y_kf[valid_set,:]

    #Get testing data
    X_kf_test=X_kf[testing_set,:]
    y_kf_test=y_kf[testing_set,:]



    #Preprocess data
    #Z-score "X_kf" inputs. 
    X_kf_train_mean=np.nanmean(X_kf_train,axis=0)
    X_kf_train_std=np.nanstd(X_kf_train,axis=0)
    X_kf_train=(X_kf_train-X_kf_train_mean)/X_kf_train_std
    X_kf_test=(X_kf_test-X_kf_train_mean)/X_kf_train_std
    X_kf_valid=(X_kf_valid-X_kf_train_mean)/X_kf_train_std

    #Zero-center outputs
    y_kf_train_mean=np.nanmean(y_kf_train,axis=0)
    y_kf_train=y_kf_train-y_kf_train_mean
    y_kf_test=y_kf_test-y_kf_train_mean
    y_kf_valid=y_kf_valid-y_kf_train_mean    




    ####### RUN KALMAN FILTER #######

    #We are going to loop through different lags, and for each lag: 
        #-we will find the optimal hyperparameter C based on the validation set R2
        #-with that hyperparameter, we will get the validation set R2 for the given lag
    #We will determine the lag as the one that gives the best validation set R2
    #Finally, using the lag and hyperparameters determined (based on above), we will get the test set R2


    #First, we set the limits of lags that we will evaluate for each dataset
    if dataset=='hc':
        valid_lags=np.arange(int(-.5/dt),int(.5/dt)) #Lags to evaluate
    if dataset=='m1':
        valid_lags=np.arange(int(-.4/dt),1) #Lags to evaluate
    if dataset=='s1':
        valid_lags=np.arange(int(-.2/dt),int(.2/dt)) #Lags to evaluate
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
            model_kf=KalmanFilterDecoder(C=C) #Define model
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
        best_params=kfBO.res['max']['max_params'] #Get the hyperparameters that give rise to the best fit
        C=best_params['C']
#         print("C=", C)

        #Get the validation set R2 using the best hyperparameters fit above:    
        model_kf=KalmanFilterDecoder(C=C) #Define model
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
    model_kf=KalmanFilterDecoder(C=C) #Define model
    model_kf.fit(X_kf_train,y_kf_train) #Fit model
    y_test_predicted_kf=model_kf.predict(X_kf_test,y_kf_test) #Get test set predictions
    #Get test set R2 values and put them in arrays
    if dataset=='hc':
        mean_r2_kf[i]=np.mean(get_R2(y_kf_test,y_test_predicted_kf)[0:2]) #Position is components 0 and 1
        print(np.mean(get_R2(y_kf_test,y_test_predicted_kf)[0:2]))
    if dataset=='m1' or dataset=='s1':
        mean_r2_kf[i]=np.mean(get_R2(y_kf_test,y_test_predicted_kf)[2:4]) #Velocity is components 2 and 3
        print(np.mean(get_R2(y_kf_test,y_test_predicted_kf)[2:4]))    



    ### Add variables to list (for saving) ###
    y_kf_test_all.append(y_kf_test)
    y_kf_valid_all.append(y_kf_valid)    
    y_kf_train_all.append(y_kf_train)    

    y_pred_kf_all.append(y_test_predicted_kf)
    y_valid_pred_kf_all.append(model_kf.predict(X_kf_valid,y_kf_valid))
    y_train_pred_kf_all.append(model_kf.predict(X_kf_train,y_kf_train))    



    ### Save ###    
    with open(save_folder+dataset+'_results_binsize_kf.pickle','wb') as f:
        pickle.dump([mean_r2_kf,y_pred_kf_all,y_valid_pred_kf_all,y_train_pred_kf_all,
                     y_kf_test_all,y_kf_valid_all,y_kf_train_all,lag],f)    


# print("time_elapsed:",time_elapsed)



# In[ ]:




