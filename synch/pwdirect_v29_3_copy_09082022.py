#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 23:13:22 2022

@author: dangoo
"""

# Stacking for y
# based on: pwdirec_v29_2.py
# Change: for v3 (sbbr), PA structure for pa_base is changed.
        # - class PA always TRUE y_true, no longer y_true for sub-train samples

# v3:
    # Try adding more prediction means to the stacks e.g. SBBR, ranking_shifting, neighbouring_ranking_estimate  
    # This version is adding SBBR
    # Result filename: stacking_y_29_2_r3.npy, stacking_y_29_2_c3.npy
    # (well the name is a bit inappropriate bc this file is stand alone now.)
    # Run from 09/07/2022 14:50
    # size 100 - 300 from qsar_data_unsorted 

    # results:
        # Direct regressions wins standard (MSE): 11.0%
        # Stacked  wins standard (MSE): 3.6%
        # Stacked  wins direct regression (MSE): 10.0%
        
        # Direct regressions wins standard (rho): 25.2%
        # Stacked wins standard (rho): 18.4%
        # Stacked wins direct regression (rho): 23.3%
        
        # Coeff examples:
            # [DirectRegressor, Sign*Abs, SBBR_c3, Sbbr_c2, SBBR_c23, 
            #  SBBR_c12, SBBR_c123] = 
            # [0.0342064	-0.00187583	 -0.320492	-0.0236645	-0.13996	
            #  0.98837	0.460915]
            # [0.0931334	-0.00981274	-0.306606	-0.0900856	0.137259	
            #  0.981796   0.181362]
    # Comment:
        # Results is bad. Why?
        ######## NOTE
        ######## BUG FOUND IN C1_keys_del
        ######## Invalidated results:
                    # stacking_y_29_2_r3.npy, stacking_y_29_2_c3.npy
                    # stacking_y_29_2_r22.npy (see pwdirect_v29_2.npy)
                    # (v0 is fine and unaffected, I think)
        ####### BUG CONTENT:
                    # for pa_base, the c1_keys_del was obtained by subtrcting
                    # c2_keys_dal and c3_keys_del from all_pairs, which is not
                    # the case in stacking base models
        ####### FIXATION:
                    # instead of subtraction, c1_keys_del are generated via
                    # permutation func + list((x,x) for x in train_ids)
        # The linear regressor is not acting like a weighted averager
        
    # Next version, v3.1: 
        # LinearRegression(positive=True)
        ##### sklearn linear regressor cannot take coef constraints
        ##### Switched to constrained_linear_regression() 
        # Result filename: stacking_y_29_3_r31.npy, stacking_y_29_3_c31.npy
        # To save time, only try the first 150 datasest. Every 10 datasets, 
        # results are printed.
        ##### 以下作废：应该把satnadard approach 也stack进去
        # Direct regressions wins standard (MSE): 17.3%
        # Stacked  wins standard (MSE): 68.7%
        # Stacked  wins direct regression (MSE): 19.3%
        
        # Direct regressions wins standard (rho): 28%
        # Stacked wins standard (rho): 56% 
        # Stacked wins direct regression (rho): 32.7%
        # In staking_y_29_3_r31.npy, rows are 
            # 1 - standard; 2 - direct regression; 3 - stacked
            
        
        # Adding more datasets (except the first 150 datasets that is already in previous file):
        ##### 以下作废：应该把satnadard approach 也stack进去
        # Filename: stacking_y_29_3_r31_2.npy & stacking_y_29_3_c31_2.npy, 
        # 158 datasets
        # run since 14/07/2022  16:40 
        # rows are 
            # 1 - standard; 1 - stacked; 2 - direct regression; 3 ~ 7 - SBBR 
        # Direct regressions wins standard (MSE): 8.9%
        # Stacked  wins standard (MSE): 16.4%
        # Stacked  wins direct regression (MSE): 81.6%
        
        # Direct regressions wins standard (rho): 20.2%
        # Stacked wins standard (rho): 20.3%
        # Stacked wins direct regression (rho): 45.0%
        
        
    # v3.5:
        # stack standard regression as well
        # 150 datasets
        # Filename: stacking_y_29_3_r35.npy & stacking_y_29_3_c35.npy
        # In staking_y_29_3_r35.npy, rows are 
            # 0 - stack; 1 - direct regression; 2 - abs*sign; 3-7 - SBBR; 8 - standard 
        # standard approach took majority of weighting most of the time
        # Stacked wins standard (MSE): 44.7%
        # Stacked wins standard (rho): 48.7%
        
        # Stacked  wins direct regression (MSE): 83%
        # Stacked  wins direct regression (Rho): 67%
        
    #v4:
        # x_meta includes sample fingerprints
        # use RFR as meta model
        # excluding abs * sign, so rows are 
            # 0 - stack; 1 - direct regression; 2 - 6  - SBBR; 7 - standard
        
        # Filename: stacking_y_29_3_r4.npy  (first 150 datasets)
        # Stacked wins standard (MSE): 12.7%
        # Stacked wins standard (rho): 16%
        
        # Stacked  wins direct regression (MSE): 48.7%
        # Stacked  wins direct regression (rho): 30.7%
        
        # With fp, the performance is much reduced
        # but mean(mse(stack) - mse(direct regression)) = -0.0607
        # but but mean(rho(stack) - rho(direct regression)) = -0.0178
        # and mean(mse(stack) - mse(standard)) = 0.0184
        
    #v4 running on datasets of size between 1000 - 1500 
    # result filename: stacking_y_29_3_r4_l.npy 
    # very slow - ~15hr per datasets
    #
    
    
    #v4 re-run 
    #added feature:
        # when fold%cv != 0, the left our test samples are now added to the 
        # last set of CV
    # Result Filename: stacking_y_29_3_r4_rerun.npy  
    # 0 - stack; 1 - direct regression; 2 - 6  - SBBR; 7 - standard; 
    # 8 - x_meta[:, 7:]; 9 - repeated SA

    
        
        
        
#necessary module
from pa_basics.import_chembl_data import dataset
from pa_basics.all_pairs import paired_data


#python package
import numpy as np
import os

from itertools import permutations,product
from scipy.stats import spearmanr, kendalltau
from random import randint
import time
from trueskill import Rating as ts_rating, rate_1vs1
from ScoreBasedTrueSkill.score_based_bayesian_rating import ScoreBasedBayesianRating as SBBR
from ScoreBasedTrueSkill.rating import Rating as sbbr_rating

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor
from sklearn.svm import SVR

from scipy.optimize import minimize



def data_check(filename):
    shuffle_state = randint(0,100)
    train_test = dataset(filename, shuffle_state = shuffle_state)
    sample_size = np.shape(train_test)[0]
    if sample_size > 300 or sample_size < 100: return True
    
    my_dict = {i:list(train_test[:,0]).count(i) for i in list(train_test[:,0])}
    max_repetition = max(my_dict.values())
    if max_repetition > 0.15 * sample_size: return True
    
    return train_test


# v1 specific used:
# def train_test_ids(train_test, cv, fold):
#     sample_size = np.shape(train_test)[0]
#     ##### stacking specific
#     ntest = int(sample_size / fold)
#     ntest_with_inner = int(sample_size / fold * (fold - 1) / fold)
#     train_ids = list(range(sample_size))
#     if cv < (fold - 1):
#         test_ids = list(range((cv)*ntest, (cv+1)*ntest + ntest_with_inner))
#         del train_ids[(cv*ntest) : ((cv+1)*ntest + ntest_with_inner)]
#     if cv == (fold - 1):
#         test_ids = list(range(cv*ntest, (cv+1)*ntest)) + list(range(ntest_with_inner))
#         del train_ids[(cv*ntest):]
#         del train_ids[:ntest_with_inner]
    
#     return train_ids, test_ids


# v2 specific:
def train_test_ids(train_test, cv, fold):
    
    sample_size = np.shape(train_test)[0]
    ntest = int(sample_size / fold) 
    train_ids = list(range(sample_size))
    if cv == fold - 1:
        start = sample_size - (sample_size - ntest*cv)
        test_ids = list(range(start, sample_size))
        del train_ids[start : sample_size]
    else:
        test_ids = list(range(cv*ntest, (cv+1)*ntest))
        del train_ids[(cv*ntest) : ((cv+1)*ntest)]
    return train_ids, test_ids



def SA_training(train_test, train_ids, test_ids, mr):
    train_samples, test_samples = np.array(train_test[train_ids]), np.array(train_test[test_ids])

    mr = RandomForestRegressor()
    x_train = train_samples[:,1:]
    y_train = train_samples[:,0]
    fitted_model = mr.fit(x_train, y_train)
    
    x_test = test_samples[:,1:]
    y_test = test_samples[:,0]
    y_pred_tr = fitted_model.predict(x_train)
    y_pred_ts = fitted_model.predict(x_test)
    # y_all_pred = np.insert(y_train, test_ids[0], y_pred_ts)
    
    return y_pred_tr, y_train, y_pred_ts, y_test



def SA_evaluation(train_test, train_ids, test_ids, y_true, mr):
    _, _, y_pred_ts, y_test = SA_training(train_test, train_ids, test_ids, mr)
    
    mse = mean_squared_error(y_test, y_pred_ts)
    mae = mean_absolute_error(y_test, y_pred_ts)
    r2 = r2_score(y_test, y_pred_ts)
    rho = spearmanr(y_test, y_pred_ts)[0] 
    ndcg = ndcg_score([y_test], [y_pred_ts])
    tau = kendalltau(y_test, y_pred_ts)[0]
    return (mse, mae, r2, rho, ndcg, tau)



class PA:
    def __init__(self, pairs, train_ids, test_ids, y_true):
        pw_train_test = self.train_test_split(pairs, train_ids, test_ids)
        train_pairs, c1_keys_del, c2_test_pairs, c2_keys_del, c3_test_pairs, c3_keys_del = pw_train_test
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.y_true = y_true
        
        self.train_pairs = train_pairs
        self.c1_keys_del = c1_keys_del
        
        self.c2_test_pairs = c2_test_pairs
        self.c2_keys_del = c2_keys_del
        
        self.c3_test_pairs = c3_test_pairs
        self.c3_keys_del = c3_keys_del
        
        self.X_train = self.train_pairs[:,1:]
        self.Y_train = self.train_pairs[:,0]
        
        self.X_test_c2 = self.c2_test_pairs[:,1:]
        self.Y_test_c2 = self.c2_test_pairs[:,0]
        self.X_test_c3 = self.c3_test_pairs[:,1:]
        self.Y_test_c3 = self.c3_test_pairs[:,0]
        
        self.c12_keys_del = self.c2_keys_del + self.c1_keys_del
        self.c23_keys_del = self.c2_keys_del + self.c3_keys_del
        self.c123_keys_del = self.c1_keys_del + self.c2_keys_del + self.c3_keys_del
        
    
    def train_test_split(self, pairs, train_ids, test_ids):
        
        def pair_test_w_train(train_ids, test_ids):
            c2test_combs= []
            for comb in product(test_ids, train_ids):
                    c2test_combs.append(comb)
                    c2test_combs.append(comb[::-1])
            return c2test_combs
    
        train_pairs = dict(pairs)
        c2_test_pairs = []
        c3_test_pairs = []
        c2_keys_del = pair_test_w_train(train_ids, test_ids)
        c3_keys_del = list(permutations(test_ids, 2)) + [(a,a) for a in test_ids]
        c1_train_pairs = []
        c1_keys_del = list(permutations(train_ids, 2)) + [(a,a) for a in train_ids]

        for key in c2_keys_del:
            c2_test_pairs.append(train_pairs.pop(key))
        for key in c3_keys_del:
            c3_test_pairs.append(train_pairs.pop(key))
        c2_test_pairs = np.array(c2_test_pairs)
        c3_test_pairs = np.array(c3_test_pairs)
        
        for key in c1_keys_del:
            c1_train_pairs.append(train_pairs.pop(key))
        c1_train_pairs = np.array(c1_train_pairs)   
        # c1_keys_del, trainp = [], []
        # for a,b in train_pairs.items():
        #     c1_keys_del.append(a)
        #     trainp.append(b)
        # train_pairs = np.array(trainp)
        
        
        return c1_train_pairs, c1_keys_del, c2_test_pairs, c2_keys_del, c3_test_pairs, c3_keys_del




    def PA_original(self,mr, pc = 1):
        if pc < 1:
            fitted_model1 = mr.fit(self.X_train[:int(pc*len(self.X_train)),:], 
                                   self.Y_train[:int(pc*len(self.Y_train))])
            
            Ytr_Oc2 = fitted_model1.predict(self.X_train[int(pc*len(self.X_train)):,:])
        else: 
            fitted_model1 = mr.fit(self.X_train, self.Y_train)
            Ytr_Oc2 = fitted_model1.predict(self.X_train)
            
        Yts_Oc2 = fitted_model1.predict(self.X_test_c2)
        Yts_Oc3 = fitted_model1.predict(self.X_test_c3)
        
        return Ytr_Oc2, Yts_Oc2, Yts_Oc3
    
    
    def PA_sign(self, mc, pc = 1):
        if pc < 1:
            
            fitted_model2 = mc.fit(self.X_train[:int(pc*len(self.X_train)),:],
                                   np.sign(self.Y_train[:int(pc*len(self.Y_train))]))
            Ytr_Sc2 = fitted_model2.predict(self.X_train[int(pc*len(self.X_train)):,:])
        else:
            fitted_model2 = mc.fit(self.X_train, np.sign(self.Y_train))
            Ytr_Sc2 = fitted_model2.predict(self.X_train)
        Yts_Sc2 = fitted_model2.predict(self.X_test_c2)
        Yts_Sc3 = fitted_model2.predict(self.X_test_c3)
        
        return Ytr_Sc2, Yts_Sc2, Yts_Sc3
        
    
    def PA_abs(self, mr, pc = 1):
        if pc < 1:
            fitted_model2 = mr.fit(abs(self.X_train[:int(pc*len(self.X_train)),:]), 
                                   abs(self.Y_train[:int(pc*len(self.Y_train))]))
            Ytr_Ac2 = fitted_model2.predict(abs(self.X_train[int(pc*len(self.X_train)):,:]))
        else:
            fitted_model2 = mr.fit(self.X_train, self.Y_train)
            Ytr_Ac2 = fitted_model2.predict(self.X_train)
        Yts_Ac2 = fitted_model2.predict(abs(self.X_test_c2))
        Yts_Ac3 = fitted_model2.predict(abs(self.X_test_c3))
    
        return Ytr_Ac2, Yts_Ac2, Yts_Ac3

    
   
    def weighted_estimate(self, y_pw_lst, test_combs_lst, y_true = None, Y_prob = None):
        if y_true is None:
            y_true = self.y_true
        if Y_prob is None: #linear arithmetic
            Y_prob = np.ones((len(y_pw_lst)))
    
        records = np.zeros((len(y_true)))
        weights = np.zeros((len(y_true)))
        
        for pair in range(len(y_pw_lst)):
            ida, idb = test_combs_lst[pair]
            delta_ab = y_pw_lst[pair]
            weight = Y_prob[pair]
    
            if ida in self.test_ids:
                # (stest, strain)
                w_esti = (y_true[idb] + delta_ab) * weight
                records[ida] += w_esti 
                weights[ida] += weight
    
            elif idb in self.test_ids:
                # (strain, stest)
                w_esti = (y_true[ida] - delta_ab) * weight
                records[idb] += w_esti 
                weights[idb] += weight
        
        return np.divide(records[self.test_ids],weights[self.test_ids])
        
    # for v3      
    def PA_rank(self, Yts_Sc2, Yts_Sc3, func):
        Ytr_Sc1 = np.array(self.Y_train)
        rate_c3, rate_c3metrics = rank_evaluation(Yts_Sc3, 
                                           self.c3_keys_del, \
                                           self.train_ids, \
                                           self.test_ids, \
                                           self.y_true, \
                                           func)
            
        rate_c2, rate_c2metrics = rank_evaluation(Yts_Sc2, 
                                           self.c2_keys_del, \
                                           self.train_ids, \
                                           self.test_ids, \
                                           self.y_true, \
                                           func)
            
        Yts_Sc23 = list(Yts_Sc2) + list(Yts_Sc3)
        rate_c23, rate_c23metrics = rank_evaluation(Yts_Sc23, 
                                           self.c23_keys_del, \
                                           self.train_ids, \
                                           self.test_ids, \
                                           self.y_true, \
                                           func)
            
        Yts_Sc12 = list(Yts_Sc2) + list(Ytr_Sc1)
        rate_c12, rate_c12metrics = rank_evaluation(Yts_Sc12, 
                                           self.c12_keys_del, \
                                           self.train_ids, \
                                           self.test_ids, \
                                           self.y_true, \
                                           func)
        
        Yts_Sc123 = list(Ytr_Sc1) + list(Yts_Sc2) + list(Yts_Sc3)
        rate_c123, rate_c123metrics = rank_evaluation(Yts_Sc123, 
                                           self.c123_keys_del, \
                                           self.train_ids, \
                                           self.test_ids, \
                                           self.y_true, \
                                           func)   
        return rate_c3, rate_c3metrics, rate_c2, rate_c2metrics, \
               rate_c23, rate_c23metrics, rate_c12, rate_c12metrics, \
               rate_c123, rate_c123metrics
            
                 
    def PA_evaluation(self, mc, mr, fold = 5):
        
        _, Yts_Sc2, Yts_Sc3 =  self.PA_sign(mc)
        _, Yts_Ac2, Yts_Ac3 =  self.PA_abs(mr)
        _, Yts_Oc2, Yts_Oc3 =  self.PA_original(mr)
        
        # v1 specific:
        # outer_test_size = int(len(self.y_true)/fold) 
        # inner_test_size = len(self.test_ids)  - outer_test_size
        # ms = SVR()
        # y_Oest = self.weighted_estimate(Yts_Oc2, self.c2_keys_del)
        # y_SAest = self.weighted_estimate(Yts_Sc2 * Yts_Ac2, self.c2_keys_del)
        # X_meta = np.vstack((y_Oest[:inner_test_size], y_SAest[:inner_test_size])).T
        # meta_model = ms.fit(X_meta, self.y_true[self.test_ids][:inner_test_size])
        
        
        # # yts_Tc2 = meta_model.predict((np.vstack((y_Oest[inner_test_size:], y_SAest[inner_test_size:])).T))
        # metrics = np.empty((3, 6))
        # metrics[1,:] = np.array(evaluation(self.y_true[self.test_ids][inner_test_size:], y_Oest[inner_test_size:]))
        # metrics[2,:] = np.array(evaluation(self.y_true[self.test_ids][inner_test_size:], yts_Tc2))       
        
        # v0 specific:
        y_Oest = self.weighted_estimate(Yts_Oc2, self.c2_keys_del)
        y_SAest = self.weighted_estimate(Yts_Sc2 * Yts_Ac2, self.c2_keys_del)
        y_Mest = (y_Oest + y_SAest)/2

  
        metrics = np.empty((4, 6))
        metrics[1,:] = np.array(evaluation(self.y_true[self.test_ids], y_Oest))
        metrics[2,:] = np.array(evaluation(self.y_true[self.test_ids], y_SAest))
        metrics[3,:] = np.array(evaluation(self.y_true[self.test_ids], y_Mest))
        
        return metrics
    
    
def evaluation(y_true, y_predict):

    rho = spearmanr(y_true, y_predict, nan_policy = "omit")[0]
    ndcg = ndcg_score([y_true], [y_predict])
    mse = mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    tau = kendalltau(y_true, y_predict)[0]
    r2 = r2_score(y_true, y_predict)
    # print(rho, ndcg, tau, mse)

    return (mse , mae, r2, rho, ndcg, tau)
    


# v2 v3 specific:
def PA_stacking(train_test, train_ids, test_ids, pairs, mc, mr ):
    # split dataset of size N: 80% train, 20% test 
    # within train, 5-fold cv
    y_true = train_test[:,0]
    train_samples = train_test[train_ids]
    fold = 5
    # meta model data input: shape = (0.8N, 2)

    # x_meta = np.empty((len(train_ids), 2)) #v2.1
    # x_meta = [[] for _ in range(2)] # v2.2
    x_meta = [[] for _ in range(1+ 1  + 1)] #v3  #3.5
    y_meta = []
    
    stack_tests = [] #v4
    for cv in range(5):
        stack_train_ids_temp, stack_test_ids_temp = train_test_ids(train_samples, cv, fold)
            # stack_train_ids = list(range(int(len(train_samples)/5*4)))
            # stack_test_ids = list(range(len(train_samples)))
            # del stack_test_ids[:int(len(train_samples)/5*4)]
        stack_train_ids = np.array(train_ids)[stack_train_ids_temp]
        stack_test_ids = np.array(train_ids)[stack_test_ids_temp]
        stack_tests += list(stack_test_ids)#v4
        
        # v3.5:
        _ , _, y_pred_sa,_ = SA_training(train_test, stack_train_ids, stack_test_ids, mr)
        
        pa_base = PA(pairs, stack_train_ids, stack_test_ids, y_true)
        
        # learn Y, |Y|, sign(Y) from RFR on 80% train
        # predict Y and (sign(Y) * |Y|) on 20% train
        _, Yts_Sc2, Yts_Sc3 =  pa_base.PA_sign(mc)
        _, Yts_Ac2, Yts_Ac3 =  pa_base.PA_abs(mr)
        _, Yts_Oc2, Yts_Oc3 =  pa_base.PA_original(mr)
        # estimate y from Y and (sign(Y) * |Y|) for 20% train
        
        
        # v3 specific:
        sbbr_rankings = pa_base.PA_rank(Yts_Oc2, Yts_Oc3, rating_sbbr)
        
        # sbbr can give 5 predictions of y simultaneously
        # keep these y, repeat 4 times from step 3 for another 20% test
        y_meta+=list(y_true[stack_test_ids])
        x_meta[0]+=list(pa_base.weighted_estimate(Yts_Oc2, pa_base.c2_keys_del))
        # x_meta[1]+=list(pa_base.weighted_estimate(Yts_Sc2 * Yts_Ac2, pa_base.c2_keys_del))
        # v3 specific:
        # for i in range(5):
        #     x_meta[i + 1]+=list(sbbr_rankings[i * 2][stack_test_ids])
            
        #v3.5
        x_meta[1] += list(y_pred_sa)
        x_meta[2] += list(sbbr_rankings[8][stack_test_ids])

    x_meta = np.array(x_meta).T
    train_fp = train_test[np.array(stack_tests), 1:] #v4
    x_meta = np.concatenate((x_meta,train_fp), axis = 1) #4
    # learn meta with estimated y for the training set
    
    # ms = LinearRegression(positive = True) #v3.1, add positive
    
    # v3.1 constrained_linear_regression
    # ms = constrained_linear_regression()
    
    #meta_model = ms.fit(x_meta, y_meta) #fixed bug
    meta_model = RandomForestRegressor(n_jobs = -1).fit(x_meta, y_meta)
    #meta_model_sa = RandomForestRegressor(n_jobs = -1).fit(x_meta[:,7:], y_meta)
    #meta_model_sa2 = RandomForestRegressor(n_jobs = -1).fit(train_test[np.array(train_ids), 1:],
    #                                                       train_test[np.array(train_ids), 0])

    pa = PA(pairs, train_ids, test_ids, y_true)
    _, Yts_Sc2, Yts_Sc3 =  pa.PA_sign(mc)
    _, Yts_Ac2, Yts_Ac3 =  pa.PA_abs(mr)
    _, Yts_Oc2, Yts_Oc3 =  pa.PA_original(mr)
    

    x_meta_test = np.empty((len(test_ids), 1 + 1 + 1))

    y_Oest = pa.weighted_estimate(Yts_Oc2,pa.c2_keys_del)
    x_meta_test[:, 0] = y_Oest
    # x_meta_test[:, 1] = pa.weighted_estimate(Yts_Sc2 * Yts_Ac2, pa.c2_keys_del)

    _ , y_train, y_pred_sa_test, _ = SA_training(train_test, train_ids, test_ids, mr)
    

    # v3:
    sbbr_rankings_test = pa.PA_rank(Yts_Oc2, Yts_Oc3, rating_sbbr)
    #for i in range(5):
    #    x_meta_test[:, i + 1] = sbbr_rankings_test[i * 2][test_ids]
        
    x_meta_test[:, 1] = y_pred_sa_test
    x_meta_test[:,2] = sbbr_rankings_test[8][test_ids]

    test_fp = train_test[np.array(test_ids), 1:] #v4
    x_meta_test = np.concatenate((x_meta_test,test_fp), axis = 1) #4
    
    y_stack_pred = meta_model.predict(x_meta_test)
    
    #######test
    #y_meta_sa = meta_model_sa.predict(x_meta_test[:,7:])
    #y_meta_sa2 = meta_model_sa2.predict(test_fp)


    #V3.1_R31 & C31
    # metrics = np.empty((5, 6))
    # metrics[1,:6] = np.array(evaluation(y_true[test_ids], y_Oest))
    # metrics[2,:6] = np.array(evaluation(y_true[test_ids], y_stack_pred))
    # metrics[3,:6] = np.array(evaluation(y_true[test_ids], sbbr_rankings_test[0][test_ids]))
    # metrics[4,:6] = np.array(evaluation(y_true[test_ids], sbbr_rankings_test[8][test_ids]))

    # coeff = []
    # coeff.append(meta_model.coef_)
    
    
    
    #V3.5
    metrics = np.empty((4, 6))
    metrics[0,:] = np.array(evaluation(y_true[test_ids], y_stack_pred))
    
    for i in range(3):
        metrics[i + 1,:6] = np.array(evaluation(y_true[test_ids], x_meta_test[:,i]))
    


    #v31 v35
    # coeff = []
    # coeff.append(meta_model.coef_)
    
    return metrics
    # , coeff #v31 v35


class constrained_linear_regression():
    
    def __init__(self, positive = True, sum_one = True, coef = None):
        self.bnds = positive
        self.cons = sum_one
        self.coef_ = coef
        
        
    def fit(self, X, Y):
    
        nsamples, nfeatures = np.shape(X)
        # Define the Model
        model = lambda b, X: sum([b[i] * X[:,i] for i in range(nfeatures)])
    
        # The objective Function to minimize (least-squares regression)
        obj = lambda b, Y, X: np.sum(np.abs(Y-model(b, X))**2)
    
        # Bounds: b[0], b[1], b[2] >= 0
        
        bnds = [(0, None) for _ in range(nfeatures)] if self.bnds else None
    
        # Constraint: b[0] + b[1] + b[2] - 1 = 0
        cons = [{"type": "eq", "fun": lambda b: sum(b) - 1}] if self.bnds else ()
    
        # Initial guess for b[1], b[2], b[3]:
        xinit = np.array([1] + [0 for _ in range(nfeatures - 1)])
    
        res = minimize(obj, args=(Y, X), x0=xinit, bounds=bnds, constraints=cons)
    
        return constrained_linear_regression(self.bnds,self.cons, res.x)
        
    
    def predict(self, Xtest):
        return np.matmul( Xtest, self.coef_)


# for v3
def rating_sbbr(comparison_results_lst, test_combs_lst, y_true, train_ids):
    nsamples = len(y_true)
    ncomparisons = len(comparison_results_lst)
    mean_train = np.mean(y_true[train_ids])
    dev_train = mean_train/ 3
    beta = mean_train / 6
    ranking = [ [sbbr_rating(mean_train, dev_train, beta)] for idx in range(nsamples)]
    
    for comp_id in range(ncomparisons):
        ida, idb = test_combs_lst[comp_id]
        comp_result = comparison_results_lst[comp_id]
        SBBR([ranking[ida], ranking[idb]],[comp_result, 0]).update_skills()
    
    return np.array([i[0].mean for i in ranking])


# for v3
def rank_evaluation(dy, combs_lst, train_ids, test_ids, y_true, func):
    y0 = func(dy, combs_lst, y_true, train_ids)
    # print(y_true[test_ids], y0[test_ids])
    rho = spearmanr(y_true[test_ids], y0[test_ids], nan_policy = "omit")[0]
    ndcg = ndcg_score([y_true[test_ids]], [y0[test_ids]])
    mse = mean_squared_error(y_true[test_ids], y0[test_ids])
    mae = mean_absolute_error(y_true[test_ids], y0[test_ids])
    tau = kendalltau(y_true[test_ids], y0[test_ids])[0]
    # print(rho, ndcg, tau, mse)

    return y0, (mse , mae, rho, ndcg, tau)




def load_datasets():
    filename_lst = []
    # directory = r'/Users/dangoo/My documents/PhD Research/Pairwise Approach/RP/qsar_data_unsorted'
    directory = r'/home/iris/PA/qsar_data_unsorted'
    # directory = r'/Users/dangoo/My documents/PhD Research/Pairwise Approach/RP/qsar_data/qsar_100_100'
    # directory = r'/Users/dangoo/My documents/PhD Research/Pairwise Approach/RP/qsar_data/'

    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                f=open(os.path.join(root,file), 'r')
                filename_lst.append(os.path.join(root,file))
                f.close()
    return filename_lst



def main(filename):
    
    train_test = data_check(filename)
    if type(train_test) == type(True):
        return

    fold = 5

    
    # y_true = np.array(train_test[:,0])
    pairs = paired_data(train_test)
    r = []
    # coeffs = [] #v31 v35
    # for run, cv in itertools.product(range(runs),range(fold)): # 5-fold CV
    for cv in range(fold):
        train_ids, test_ids = train_test_ids(train_test, cv, fold)
        mr = RandomForestRegressor(n_jobs = -1)
        mc = RandomForestClassifier(n_jobs = -1)
        
        # metrics_ref = SA_evaluation(train_test, train_ids, test_ids, y_true, mr)
        # note: v0 & v1 uses:
        # pa = PA(pairs, train_ids, test_ids,y_true)
        # metrics = pa.PA_evaluation(mc, mr)
        
        # v2 uses:
        # metrics = PA_stacking(train_test, train_ids, test_ids, pairs, mc, mr)
        # v3 uses:
        metrics = PA_stacking(train_test, train_ids, test_ids, pairs, mc, mr)

        # metrics[0, :6] = np.array(metrics_ref)  # uncommented for v3.5
        # coeffs.append(coeff) #v31 v35
        r.append(metrics)
        # print(np.mean(np.array(r), axis = 0))
        
    return np.array(r)
    #, np.array(coeffs) #v31 v35



if __name__ == '__main__':

    filename_lst = load_datasets()
    r_all_dataset_D29 = []
    # r_coeff = [] #v31 v35
    for filename in filename_lst:
        if len(r_all_dataset_D29) == 150:
            break
        r = main(filename)
        if type(r) != type(None):
            r_all_dataset_D29.append(r)
            np.save("stacking_y_29_3_r4_3bases.npy", np.array(r_all_dataset_D29))
            # r_coeff.append(r[1])
            # np.save("stacking_y_29_3_c4.npy", np.array(r_coeff))
