#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:11:55 2022

@author: dangoo
"""
import numpy as np
import pandas as pd
import os
import concurrent.futures
from pa_basics.import_chembl_data import dataset



def load_datasets():
    filename_lst = []
    directory = r'/Users/dangoo/My documents/PhD Research/Pairwise Approach/RP/qsar_data_unsorted'
    # directory = r'/home/iris/PA/qsar_data_unsorted'
    # directory = r'/Users/dangoo/My documents/PhD Research/Pairwise Approach/RP/qsar_data/qsar_100_100'

    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                f=open(os.path.join(root,file), 'r')
                filename_lst.append(os.path.join(root,file))
                f.close()
    return filename_lst


def dataset_shape(filename):
    file = dataset(filename)
    return np.shape(file)


def reptition_rate(filename):
    train_test = dataset(filename)
    sample_size = np.shape(train_test)[0]    
    my_dict = {i:list(train_test[:,0]).count(i) for i in list(train_test[:,0])}
    max_repetition = max(my_dict.values())
    return max_repetition / sample_size


if __name__ == "__main__":
    # record = pd.DataFrame()
    # filename_lst = load_datasets()
    # record["File directory"] = filename_lst
    # record["File name"] = [filename[-(18 + len(filename) - 98):] for filename in filename_lst]
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     shapes = list(executor.map(dataset_shape, filename_lst))
        
    # record["Dimension(filtered)"] = shapes
    # record.to_csv("chembl_datasets_info.csv")
    
    
    record = pd.read_csv("chembl_datasets_info.csv")
    filename_lst = record["File directory"]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        reptitions = list(executor.map(reptition_rate, filename_lst))
    record["Reptition Rate"] = reptitions 
    
    