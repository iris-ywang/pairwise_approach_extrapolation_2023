#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:51:49 2022

@author: dangoo
"""

#import ChEMBL dataset


import pandas as pd
import numpy as np
from sklearn.utils import shuffle

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