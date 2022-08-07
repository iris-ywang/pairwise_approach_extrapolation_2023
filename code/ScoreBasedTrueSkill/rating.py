#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:23:04 2022

@author: dangoo
"""
from ScoreBasedTrueSkill import Gauss



# trueskill/lib/saulabs/trueskill/rating.rb 
class Rating(Gauss.Distribution):

      
    def __init__(self, mean, deviation, tau = 25/300.0,  activity = 1.0):
        super().__init__(mean = mean, deviation = deviation)
        self.activity = activity
        self.tau = tau
        
        
    # def __setattr__(self, tau):
    #     self.tau = tau
    #     self.tau_squared = self.tau**2
        
        
    @property
    def tau_squared(self):
        return self.tau ** 2

#end