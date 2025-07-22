#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:20:50 2022

@author: ivan
"""

K=32 #K is the number of bins for each dimension in the HS histogram
std_noise=[0.25, 0.25, 0.01, 0.01, 1e-2, 1e-2, 1e-3, 1e-3] #values of noise std for each parameter in the state matrix
alpha = 27.0 #exponent to increase the sharpness of the particle weight distribution
prediction = 'weighted_avg' #Method to compute the final prediction of the object state
learning_rate = 0.05  # for adaptative
update_threshold = 0.8  # similarity threshold for updating
w_color = 0.1   # peso para la similitud de color
w_texture = 0.1 # peso para la similitud LBP
w_hog = 0.8
