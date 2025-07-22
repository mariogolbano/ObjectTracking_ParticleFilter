#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:34:54 2022

@author: ivan
"""
import numpy as np
from skimage.color import gray2rgb
from skimage.util import img_as_ubyte
import cv2
import pdb

def showBB(im,bbs):
   
    if im.ndim==2:
        im=gray2rgb(im)
    H, W, channels=im.shape;
    im=img_as_ubyte(im)
    colors=np.array([[0, 255, 0],[0, 0, 255],[255, 0, 0],[ 255, 255, 0],[ 0, 255, 255],[ 255, 0, 255 ],[255, 128, 0 ],[0, 255, 128 ],[128, 255, 0 ],[ 128, 128, 128 ],[ 128, 128, 0 ],[ 128, 0, 128 ],[ 0, 128, 128]]);
    for b in range(bbs.shape[0]):
        bb=np.round(bbs[b,:]).astype(int);
        bb[2]=np.minimum(bb[0]+bb[2],W-1);
        bb[3]=np.minimum(bb[1]+bb[3],H-1);
        bb[0]=np.maximum(bb[0],0);
        bb[1]=np.maximum(bb[1],0);
        try:
            im[bb[1],bb[0]:bb[2],0]=colors[b,0];
            im[bb[1],bb[0]:bb[2],1]=colors[b,1];
            im[bb[1],bb[0]:bb[2],2]=colors[b,2];
            im[bb[3],bb[0]:bb[2],0]=colors[b,0];
            im[bb[3],bb[0]:bb[2],1]=colors[b,1];
            im[bb[3],bb[0]:bb[2],2]=colors[b,2];
            im[bb[1]:bb[3],bb[0],0]=colors[b,0];
            im[bb[1]:bb[3],bb[0],1]=colors[b,1];
            im[bb[1]:bb[3],bb[0],2]=colors[b,2];
            im[bb[1]:bb[3],bb[2],0]=colors[b,0];
            im[bb[1]:bb[3],bb[2],1]=colors[b,1];
            im[bb[1]:bb[3],bb[2],2]=colors[b,2];
        except:
            print('Bounding box is getting out of the image')
        
    return im

#Function that shows particles as points overlaid in the image (we do not consider width, height and dynamic elements in the state)
#      im = showParticles(im,x)     
#Parameters:
#       - im: the image
#       - x: The matrix with the state respresented by the particles
#Output:
#        - im: the image with overlaid particles
def showParticles(im,x):

    if im.ndim==2:
        im=gray2rgb(im)
    H, W, channels=im.shape;
    im=img_as_ubyte(im)

    xcoord=np.round(x[:,0]).astype(int);
    ycoord=np.round(x[:,1]).astype(int);
    
    xcoord=np.clip(xcoord,0,W-1)
    ycoord=np.clip(ycoord,0,H-1)
    points=np.zeros_like(im)
    points[ycoord,xcoord,:]=255
    points=cv2.dilate(points,np.ones((2,2)))
    im=np.maximum(im,points)
    # xidx=(xcoord*H+ycoord).astype(int);
    
    # for c in range(channels):
    #     imC=im[:,:,c]
    #     imC[xidx]=255
    #     im[:,:,c]=imC
    return im
