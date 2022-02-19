#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:16:58 2022

@author: dangoo
"""
import numpy as np
import pandas as pd
import os

import itertools
from itertools import permutations,combinations,product
from scipy.stats import spearmanr
from random import randint,sample
import time
import matplotlib.pyplot as plt
from sklearn.utils import resample

from sklearn.utils import shuffle
from sklearn.metrics import jaccard_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer

import concurrent.futures




############################################################
# PART I
# Import dataset from .csv and perform basic feature selection.


# Function: Import dataset from filename.csv and perform basic feature selection.
# Input: string of (directory of the dataset + filename); random state 
# Output: numpy array of QSAR data 

def dataset(filename, shuffle_state = None):
    orig_data = import_data(filename, shuffle_state)
    filter1 = uniform_features(orig_data)
    filter2 = duplicated_features(filter1)
    return filter2



# Function: to import raw data from csv to numpy array
# Input: string of (directory of the dataset + filename); random state 
# Output: a numpy array of raw data

def import_data(filename, shuffle_state):
    df = pd.read_csv(filename)
    try: 
        data = pd.DataFrame(data=df).to_numpy().astype(np.float64)
    except:
        del df['molecule_id']
        data = pd.DataFrame(data=df).to_numpy().astype(np.float64)
        
    if shuffle_state != None:
        data = shuffle(data, random_state = shuffle_state)
    return data



# Function: to remove the all-zeros and all-ones features from the dataset
# Input: numpy array
# Output: numpy array 

def uniform_features(data):
    nsamples, ncolumns = np.shape(data)
    id_lst = []
    for fid in range(1, ncolumns):
        if np.all(data[:,fid] == data[0,fid]):
            id_lst.append(fid)
    data = np.delete(data, id_lst, axis = 1)
    return data



# Function: to remove the duplicated features from the dataset
# Input: numpy array
# Output: numpy array 

def duplicated_features(data):
    y = data[:,0:1]
    a = data[:, 1:].T
    order = np.lexsort(a.T)
    a = a[order]

    diff = np.diff(a, axis = 0)
    ui  = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    new_train = a[ui].T
    new_train_test = np.concatenate((y, new_train), axis = 1)
    return new_train_test



############################################################
# PART II
# Making pairs from QSAR datasets.



# Function: Generate all possible pairs from a QSAR dataset
# Input: numpy array of a QSAR dataset
# Output: dictionary 
#         key = (drugA_id, drugB_id); value = [y_pair, x_pair1, x_pair2...]
 
def paired_data(data):
    nsamples, ncolumns = np.shape(data)
    
    # find all the pairs by index of samples
    perm_pairs = list(permutations(range(nsamples),2)) + [(a,a) for a in range(nsamples)]
    ncombs = len(perm_pairs)
    d = {}
    for pairid in range(ncombs):
        sid_a, sid_b = perm_pairs[pairid]
        sample_a = data[sid_a : sid_a+1, :]
        sample_b = data[sid_b : sid_b+1, :]
        
        pair_ab = pair_2samples(ncolumns, sample_a, sample_b)
        d[perm_pairs[pairid]] = pair_ab
    return d



# Function: Transform the information from two single samples to a pair
# Input:  integer number of columns; 
#         numpy array of Sample A in the shape of (1, ncolumns)
#         numpy array of Sample B in the shape of (1, ncolumns)
# Output: list of Pair AB
#     
#       Note the Rules of pairwise features:
#           x_A = 1 & x_B = 1 -> X_AB = 2
#           x_A = 1 & x_B = 0 -> X_AB = 1
#           x_A = 0 & x_B = 1 -> X_AB = -1
#           x_A = 1 & x_B = 0 -> X_AB = 0

def pair_2samples(ncolumns, sample_a, sample_b):
     delta_y = sample_a[0,0] - sample_b[0,0]
     new_sample = [delta_y]

     for fid in range(1, ncolumns):
         f_value_a = sample_a[0, fid]
         f_value_b = sample_b[0, fid]

         if f_value_a == f_value_b:
             assign = f_value_a + f_value_b
             new_sample.append(assign)

         elif f_value_a != f_value_b:
             assign = f_value_a - f_value_b
             new_sample.append(assign)
     return new_sample



# Function: Find C2-type test pairs by sample IDs
#            i.e. find keys as (drugA_id, drugB_id)
# Input: list of train drugs' IDs;
#        list of test drugs' IDs
# Output: list of C2-type test pairs IDs in form of (drugA_id, drugB_id)
    
def pair_test_w_train(train_ids, test_ids):
    c2test_combs= []
    for comb in product(test_ids, train_ids):
            c2test_combs.append(comb)
            c2test_combs.append(comb[::-1])
    return c2test_combs



# Function: Generate train pairs and test pairs according to training drug IDs
#           and test drug IDs
#
# Input: dictionary of all the possible pairs;
#        list of train drugs' IDs;
#        list of test drugs' IDs
# Output: numpy array of train pairs; numpy array of test pairs

def train_test_split(pairs, train_ids, test_ids):
    train_pairs = dict(pairs)
    test_pairs = []
    test_pair_ids = pair_test_w_train(train_ids, test_ids) 

    for key in test_pair_ids: #Remove C2 test pairs from all pairs
        test_pairs.append(train_pairs.pop(key))
    for key in list(permutations(test_ids, 2)): #Remove C3 test pairs from all pairs
        train_pairs.pop(key)
    test_pairs = np.array(test_pairs)

    trainp = []
    for a,b in train_pairs.items():
        trainp.append(b)
    train_pairs = np.array(trainp)

    return train_pairs, test_pairs, test_pair_ids



############################################################
# PART IV Standard Approach

# Function: Run standard approach
#
# Input: numpy array of train drugs samples;
#        numpy array of test drugs samples;
# Output: numpy array of errors in pairwise differences(Y)
#         numpy array of errors in estiamted activity(y)

def standard_approach(train_samples, test_samples):
    ntrain = np.shape(train_samples)[0]
    ntest = np.shape(test_samples)[0]
    
    m = RandomForestRegressor(n_jobs = -1)
    x_train = train_samples[:,1:]
    y_train = train_samples[:,0]
    fitted_model = m.fit(x_train, y_train)
    
    x_test = test_samples[:,1:]
    y_test = test_samples[:,0]
    y_pred_tr = fitted_model.predict(x_train)
    y_pred_ts = fitted_model.predict(x_test)

    msetr, maetr, r2tr = pairwise_differences(ntrain, y_train, y_pred_tr, ntrain, y_train, trainpd = True)
    msets, maets, r2ts = pairwise_differences(ntest,  y_test,  y_pred_ts, ntrain, y_train)
    errors = np.array([[msetr, maetr, r2tr], 
                       [msets, maets, r2ts]])
    
    mse_ref = mean_squared_error(y_test, y_pred_ts)
    rhoref = spearmanr(y_pred_ts, y_test)[0]
    return errors, mse_ref,rhoref



'''
Function: Calculate the error in C2-type pairwise differences or training 
          pairwise difference from predicted activities.
Input:  number of drug samples being predicted
        numpy array of true values of drug samples being predicted
        numpy array of predicted values of drug samples being predicted
        number of training drugs
        numpy array of true values of trainig drugs (for pairing)
        trainpd = True: calculate pairwise training error
        trainpd = False: calculate C2 type pairwise error
Output: mean squared error (float)
        spearman coefficient (float)
        R2(float)
'''

def pairwise_differences(nts, ytstrue, ytspred, ntr, ytrtrue, trainpd = False):
     diff_true, diff_pred = [],[]
     for train, test in itertools.product(range(ntr), range(nts)):
         
         diff_pred.append( ytrtrue[train] - ytspred[test])
         if trainpd == False: diff_pred.append( - ytrtrue[train] + ytspred[test])
         
         diff_true.append( ytrtrue[train] - ytstrue[test])
         if trainpd == False: diff_true.append( -ytrtrue[train] + ytstrue[test])
     
     mse = mean_squared_error(diff_true, diff_pred)
     mae = mean_absolute_error(diff_true, diff_pred)
     r2 = r2_score(diff_true, diff_pred)
     return mse, mae, r2
 
    
 
############################################################
# PART V Pairwise Approach


# Function: Run pairwise approach
# Input:  dictionary of all the possible pairs;
#         list of train drug IDs
#         list of test pairs IDs
#         numpy array of drug
#         numpy array of true activity values for all drugs (1D)
# Output: numpy array of errors in pairwise differences(Y)
#         numpy array of errors in estiamted activity(y)


def pairwise_approach(pairs, train_ids, test_ids, y_true):
    pairs = dict(pairs)
    pw_train_test = train_test_split(pairs, train_ids, test_ids)
    train_pairs, test_pairs, c2test_ids = pw_train_test

    X_train = train_pairs[:,1:]
    Y_train = train_pairs[:,0]
    X_test = test_pairs[:,1:]
    Y_test = test_pairs[:,0]
    
    Ytr_IRF, Yts_IRF = IRF(train_ids, pairs, X_train, X_test)
    
    errors = np.empty((4,3))
    mse = mean_squared_error(Y_train, Ytr_IRF)
    mae = mean_absolute_error(Y_train, Ytr_IRF)
    r2 = r2_score(Y_train, Ytr_IRF)
    errors[0,:] = mse, mae,r2

    mse = mean_squared_error(Y_test, Yts_IRF)
    mae = mean_absolute_error(Y_test, Yts_IRF)
    r2 = r2_score(Y_test, Yts_IRF)
    errors[1,:] = mse, mae,r2
    
    m = RandomForestRegressor(n_jobs = -1)
    fitted_model = m.fit(X_train, Y_train)
    Ytr_RF = fitted_model.predict(X_train)
    Yts_RF = fitted_model.predict(X_test)

    mse = mean_squared_error(Y_train, Ytr_RF)
    mae = mean_absolute_error(Y_train, Ytr_RF)
    r2 = r2_score(Y_train, Ytr_RF)
    errors[2,:] = mse, mae,r2

    mse = mean_squared_error(Y_test, Yts_RF)
    mae = mean_absolute_error(Y_test, Yts_RF)
    r2 = r2_score(Y_test, Yts_RF)
    errors[3,:] = mse, mae,r2
    
    y_pred_IRF = weighted_average(Yts_IRF, c2test_ids, y_true, test_ids)
    y_pred_RF = weighted_average(Yts_RF, c2test_ids, y_true, test_ids)
    
    results = [mean_squared_error(y_true[test_ids], y_pred_IRF),
               spearmanr(y_true[test_ids], y_pred_IRF)[0],
               mean_squared_error(y_true[test_ids], y_pred_RF),
               spearmanr(y_true[test_ids], y_pred_RF)[0], ]
    return errors, results



# Function: Build forest from decision trees, each of which is fed with 
#           non-overlapping training pairs
# Input: list of training drugs IDs
#        dictionary of all the pairs
#        numpy array of X(train pairs)
#        numpy array of X(test pairs)
# Output: numpy array of predicted Y(train pair)
#         numpy array of predicted Y(test pair)
#
# Note: X-features; Y-differences in activities
def IRF(train_ids, pairs, X_train, X_test):
    ntrees = 200 * len(train_ids)
    trees = []
    while len(trees) < ntrees:
        tree_model = DecisionTreeRegressor(n_jobs = -1)

        trpair_ids = indep_subdataset(train_ids)
        subtr_pairs = np.array(list(map(pairs.get, trpair_ids)))
        X_subtr = subtr_pairs[:,1:]
        Y_subtr = subtr_pairs[:,0]
        tree_model.fit(X_subtr, Y_subtr)
        trees.append(tree_model)

    Ytr_pred = [tree.predict(X_train) for tree in trees]
    Ytr_pred = np.mean(np.array(Ytr_pred), axis = 0)
    
    Yts_pred = [tree.predict(X_test) for tree in trees]
    Yts_pred = np.mean(np.array(Yts_pred), axis = 0)
    
    return Ytr_pred, Yts_pred


# Function: Randomly generate a list of train pair IDs with non-overlaping 
#           training drugs
# Input: list of training drugs IDs
# Output: list of train pair IDs

def indep_subdataset(train_ids):
    remain = list(train_ids)
    trpairs = []
    while len(remain) > 1:
        a,b = sample(remain, 2)
        trpairs.append((a,b))
        remain.remove(a)
        remain.remove(b)
    return trpairs




# Function: Calculated weighted average of activity values from pairwise differences 
# Input:  numpy array of predicted pairwise differences (1D)
#         list of test pairs IDs
#         numpy array of predicted Y(test pair) (1D)
#         numpy array of true activity values for all drugs (1D)
#         list of test drug IDs
#         Y_prob = None: no weight information is given, i.e. linear average
# Output: numpy array of estimated activity values


def weighted_average(Y_pred, test_combs_lst, y_true, test_ids, Y_prob = None):
    if Y_prob is None: #linear arithmetic
        Y_prob = np.ones((len(Y_pred)))

    records = np.zeros((len(test_ids)))
    weights = np.zeros((len(test_ids)))

    for pair in range(len(Y_pred)):
        ida, idb = test_combs_lst[pair]
        delta_ab = Y_pred[pair]
        weight = Y_prob[pair]

        if ida in test_ids:
            # (stest, strain)
            w_esti = (y_true[idb] + delta_ab) * weight
            records[ida - min(test_ids)] += w_esti
            weights[ida - min(test_ids)] += weight

        elif idb in test_ids:
            # (strain, stest)
            w_esti = (y_true[ida] - delta_ab) * weight
            records[idb - min(test_ids)] += w_esti
            weights[idb - min(test_ids)] += weight

    return np.divide(records,weights)                               




############################################################
# PART VI Main



# Function: Calculated weighted average of activity values from pairwise differences 
# Input:  None 
#        (but you need to change the datasets directory in your computer)
# Output: list of (directory of the dataset + filename)

def load_datasets():
    filename_lst = []
    # directory = r'/Users/dangoo/My documents/PhD Research/Pairwise Approach/RP/qsar_data/qsar_100_100/'
    directory = r'/home/iris/PA/qsar_data_unsorted'
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                f=open(os.path.join(root,file), 'r')
                filename_lst.append(os.path.join(root,file))
                f.close()
    return filename_lst



# Function: Check if a dataset is too large or contains too many (15%) repeated 
#           activity values
# Input:  numpy array of QSAR data 
# Output: True/False

def data_check(train_test):
    sample_size = np.shape(train_test)[0]
    if sample_size > 500:
        return True
    
    my_dict = {i:list(train_test[:,0]).count(i) for i in list(train_test[:,0])}
    max_repetition = max(my_dict.values())
    if max_repetition > 0.15 * sample_size:
        return True
    
    
    
    
def main(filename):
    shuffle_state = randint(0,100)
    # load dataset
    train_test = dataset(filename, shuffle_state = shuffle_state)
    
    if data_check(train_test):
        return
    
    sample_size = np.shape(train_test)[0]
    ntest = int(sample_size / 5) 
    y_true = np.array(train_test[:,0])
    pairs = paired_data(train_test)
    
    metrics = np.empty((3,5,6)) #NOTE!
    for run in range(3): # repeat runs
        for cv in range(5): # 5-fold CV
            test_ids = list(range(cv*ntest, (cv+1)*ntest))
            train_ids = list(range(sample_size))
            del train_ids[(cv*ntest) : ((cv+1)*ntest)]
            
            train_samples, test_samples = np.array(train_test[train_ids]), np.array(train_test[test_ids])
            es, mseref,rhoref = standard_approach(train_samples, test_samples)
            
            ep, output = pairwise_approach(pairs, train_ids,test_ids, y_true)

            mseirf, rhoirf, mserf,rhorf = output
            metrics[run, cv,:] = mseref, mseirf, mserf, rhoref, rhoirf, rhorf
            errors = np.concatenate((es,ep))
            print(errors)
    # print(np.mean(metrics,axis = (0,1)))
    # print(filename, "\n")
    np.save("middle_results/"+str(filename)[-8:],metrics)
    return metrics



if __name__ == '__main__':

    filename_lst = load_datasets()[:500]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results_mse = []
        results = executor.map(main, filename_lst)

        for result in results:
            if type(result) == np.ndarray:
                results_mse.append(result)

        np.save("IRF.npy",results_mse)
    print(filename_lst)
