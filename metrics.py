#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:04:29 2022

@author: ivan
"""
import numpy as np
import pdb

def computeJI(gtbox, predbox):

    gtbox[2:]=gtbox[:2]+gtbox[2:];
    predbox[2:]=predbox[:2]+predbox[2:];
    
    x_left = np.maximum(gtbox[0], predbox[0]);
    y_top = np.maximum(gtbox[1],  predbox[1]);
    x_right = np.minimum(gtbox[2], predbox[2]);
    y_bottom = np.minimum(gtbox[3], predbox[3]);

    #if they do not intersect
    if x_right < x_left or y_bottom < y_top:
        JI=0
    else:
        #Intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top);
        
        #Union areas
        gt_area = (gtbox[2] - gtbox[0]) * (gtbox[3] - gtbox[1]);
        pred_area = (predbox[2] - predbox[0]) * (predbox[3] - predbox[1]);
        JI=intersection_area/(gt_area+pred_area-intersection_area);
    return JI
