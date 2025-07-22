#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:06:37 2022

@author: ivan
"""
import numpy as np
from skimage.color import rgb2hsv
import pdb
import numpy.random as npr
import config as cfg
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.feature import hog
import cv2



def computeHSVHistograms(im,K):
    
    #Convert image to HSV and double precision
    im =rgb2hsv(im);
    #im = im2double(im);
    h,w,c = im.shape
    #Vectorize image
    im =np.reshape(im,(h*w,c))
    #Only H and S (remove Value as it does not provide valuable information for tracking)
    im=im[:,:2]
    
    #Quantize the values
    r=np.floor((im-1e-30)*K);

    rlin=r[:,0]*K+r[:,1];
    
    hist,edges = np.histogram(rlin,bins=range(0,K*K))
    
        
    hist=hist/(hist.sum()+1e-10);
    
    return hist

def computeSpatialHSVHistograms(im, K, grid=(2,2)):
    """
    Divide la imagen im en una cuadrícula definida por grid (filas, columnas)
    y calcula el histograma HSV (usando computeHSVHistograms) para cada celda.
    Retorna una lista con los histogramas de cada celda.
    """
    h, w, _ = im.shape
    n_rows, n_cols = grid
    cell_h = h // n_rows
    cell_w = w // n_cols
    spatial_hists = []
    for i in range(n_rows):
        for j in range(n_cols):
            # Extraer la celda correspondiente
            y0 = i * cell_h
            y1 = (i + 1) * cell_h if i < n_rows - 1 else h
            x0 = j * cell_w
            x1 = (j + 1) * cell_w if j < n_cols - 1 else w
            cell = im[y0:y1, x0:x1, :]
            # Calcular y almacenar el histograma para la celda
            hist_cell = computeHSVHistograms(cell, K)
            spatial_hists.append(hist_cell)
    return spatial_hists

def computeLBPHistogram(im, numPoints=24, radius=3, bins=256):
    # Convertir la imagen a escala de grises
    gray = rgb2gray(im)
    lbp = local_binary_pattern(gray, numPoints, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=bins, range=(0, bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-10)
    return hist

def computeHOGDescriptor(im, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    # Convertir a escala de grises
    gray = rgb2gray(im)
    hog_descriptor, _ = hog(gray, orientations=orientations,
                            pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block,
                            block_norm='L2-Hys',
                            visualize=True)
    return hog_descriptor

def computeHOGDescriptor_cv2(im, winSize=None, blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9):
    # Si no se especifica winSize, usar el tamaño de la imagen.
    if winSize is None:
        winSize = (im.shape[1], im.shape[0])
    # Ajustar winSize para que se cumpla la condición
    new_w = ((winSize[0] - blockSize[0]) // blockStride[0]) * blockStride[0] + blockSize[0]
    new_h = ((winSize[1] - blockSize[1]) // blockStride[1]) * blockStride[1] + blockSize[1]
    winSize = (new_w, new_h)
    
    # Convertir la imagen a uint8 si no lo es
    if im.dtype != np.uint8:
        if im.max() <= 1:
            im_uint8 = (im * 255).astype(np.uint8)
        else:
            im_uint8 = im.astype(np.uint8)
    else:
        im_uint8 = im
        
    # Convertir a escala de grises
    if len(im_uint8.shape) == 3:
        gray = cv2.cvtColor(im_uint8, cv2.COLOR_BGR2GRAY)
    else:
        gray = im_uint8

    # Crear el descriptor HOG de OpenCV
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    descriptor = hog.compute(gray)
    return descriptor.flatten()


class particle_filter:
    
    def __init__(self, im0, bbox,numParticles=100,step=1):
        
        self.K = cfg.K #K is the number of bins in the histogram
        self.N = numParticles #Number of particles
        self.t = step #Delay between frames for the state-transition matrix
        self.alpha = cfg.alpha #exponent to increase the sharpness of the particle weight distribution
        self.w_color = cfg.w_color
        self.w_texture = cfg.w_texture
        self.w_hog = cfg.w_hog
        
        #Set the initial state X_init=[bb_center_x bb_center_y bb_width  bb_height velocity_x velocity_y velocity_width velocity_height]
        xstatic=np.array([bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3], bbox[2], bbox[3]])
        xdynamic=np.zeros((4,))
        self.x_init = np.concatenate((xstatic, xdynamic),axis=0)

        #State-transition matrix: constant velocity model
        #A=[np.eye(4) t*eye(4); zeros(4) eye(4)];   
        self.A=np.block([[np.eye(4), self.t*np.eye(4)],[np.zeros((4,4)), np.eye(4)]]);   
                    
        #We obtain the visual representation of the original object
        self.bbox = np.round(bbox);
        objim = im0[self.bbox[1]:self.bbox[1]+self.bbox[3], self.bbox[0]:self.bbox[0]+self.bbox[2],:]
        self.ref_size = (objim.shape[1], objim.shape[0])  # (ancho, alto)

        #Compute the reference histogram => work in hsv space
        self.hist_ref =  computeHSVHistograms(objim, self.K)  
        
        #compute spatial reference histogram
        #self.hist_ref_spatial = computeSpatialHSVHistograms(objim, self.K, grid=(2,2))

        #compute lbp (texture) histogram        
        self.lbp_ref = computeLBPHistogram(objim)

        #compute hog descriptor skimage
        #self.hog_ref = computeHOGDescriptor(objim)
        
        #compute hog descriptor cv2
        candidate_resized = cv2.resize(objim, (objim.shape[1], objim.shape[0]))  # O directamente usar objim si ya es el tamaño de referencia
        self.hog_ref = computeHOGDescriptor_cv2(candidate_resized, winSize=self.ref_size)


        #Copy the state to all particles x is NxP being N the number of particles and P the number of parameters 
        self.x=np.tile(self.x_init,(self.N,1));

        #Initialize weights uniformly
        self.w=(1/self.N)*np.ones((self.N,));
        #Cumulative weights for particle resampling
        self.c=np.cumsum(self.w);

        #Vector with standard deviations of additive gaussian noise
        #Each dimension corresponds with one element in the state
        self.Sigma=np.array(cfg.std_noise).transpose(); 
        #Make sigma of static variables proportional to bounding box size
        self.Sigma[:2]=self.Sigma[:2]*np.min(self.bbox[2:4]);
        self.Sigma[2:4]=self.Sigma[2:4]*self.bbox[2:4];


    def update(self, im):
        
        #Number of params in the state
        P = self.x.shape[1] 
        #Generate new data
        x_new = self.x.copy();
      
        
        #Dimensions of the frame
        height, width, colors=im.shape; 
        idx_particles=np.zeros((self.N,));
        
        #Particles loop
        for i in range(self.N):
            #####STEP 1: PARTICLE RESAMPLING#####
            val=npr.rand();
            #Choose the particle
            idx_particle = np.where(self.c>val)[0][0];
            idx_particles[idx_particle]+=1;
            #Get the state of the particle
            z=self.x[idx_particle,:]
            
            #####STEP 2: UPDATE THE PARTICLE STATE#######
            #Generate noise
            #Matrix with noise
            noise=self.Sigma*npr.randn(P);
            x_new[i,:]=(self.A@z+noise);

            #####STEP 3: EXTRACT THE CANDIDATE AREA########
            try:
                #We extract the region in the image corresponding with the bounding box
                limy=np.array([np.ceil(x_new[i,1]-0.5*x_new[i,3]), np.floor(x_new[i,1]+0.5*x_new[i,3])],dtype=int);
                limy=np.clip(limy,0,height);
                limx=np.array([np.ceil(x_new[i,0]-0.5*x_new[i,2]), np.floor(x_new[i,0]+0.5*x_new[i,2])],dtype=int);
                limx=np.clip(limx,0,width);
                candidate_reg=im[limy[0]:limy[1],limx[0]:limx[1],:]; 
            except:
                
                candidate_reg = np.zeros_like(self.hist_ref);
            
        
            ###########STEP 4: COMPUTE THE COLOR HISTOGRAM###########
            hist=computeHSVHistograms(candidate_reg,self.K);

            # Spatial histogram 
            #candidate_spatial = computeSpatialHSVHistograms(candidate_reg, self.K, grid=(2,2))

            #texture histogram
            lbp_candidate = computeLBPHistogram(candidate_reg)

            # first implementation of hog
            #candidate_resized = cv2.resize(candidate_reg, self.ref_size)
            #hog_candidate = computeHOGDescriptor(candidate_resized)

            # hog with cv2
            candidate_resized = cv2.resize(candidate_reg, self.ref_size)
            hog_candidate = computeHOGDescriptor_cv2(candidate_resized, winSize=self.ref_size)
            ###########STEP 5: COMPUTE THE BATTACHARYYA COEFICCIENT BETWEEN HISTOGRAMS###########
            #hist_intersect=(np.sqrt(self.hist_ref*hist)).sum();

            #Combination of hist_intersect and spatil histogram
            ###########STEP 5: COMPUTE THE SIMILARITY MEASURE###########
            # Medida global (Bhattacharyya) con el histograma global
            sim_global = (np.sqrt(self.hist_ref * hist)).sum()

            #similarity of texture with bachayyarta
            sim_lbp = (np.sqrt(self.lbp_ref * lbp_candidate)).sum()

            hog_distance = np.linalg.norm(self.hog_ref - hog_candidate)
            sim_hog = np.exp(-hog_distance)  # convertir distancia en similitud

            # Medida espacial: calcular la similitud para cada celda y promediar
            #sim_total = 0.0
            #n_cells = len(candidate_spatial)  # Debe ser 4 para grid=(2,2)
            #for idx in range(n_cells):
            #    sim_cell = (np.sqrt(self.hist_ref_spatial[idx] * candidate_spatial[idx])).sum()
            #    sim_total += sim_cell
            #sim_spatial = sim_total / n_cells

            # Combinar ambas medidas (por ejemplo, promedio simple)
            #sim_combined = (sim_global + sim_spatial) / 2.0

            # fuse all similarities
            final_similarity = self.w_color * sim_global + self.w_texture * sim_lbp + self.w_hog * sim_hog
            self.w[i] = final_similarity ** self.alpha


            
            #Update the weight of the particle
            #Alpha controls how sharp are the weights
            #self.w[i] = hist_intersect**self.alpha;   
        
        
        ###########STEP 6: UPDATE X, NORMALIZE THE WEIGHTS AND RECOMPUTE C###########
        self.x=x_new;
        if self.w.sum()>1e-30:
           self.w=self.w/self.w.sum();
        else:
            print('Error');
            self.w[...]=1/self.N;
        
        self.c=np.cumsum(self.w);
        
        ###########STEP 7: ESTIMATE THE BOUNDING BOX FROM THE PARTICLES###########
        # Weighted average
        if cfg.prediction=='weighted_avg':
            x_global=np.sum(self.w[:,np.newaxis]*self.x,axis=0)
        #Best particle
        elif cfg.prediction=='max':
            idx_particle=np.argmax(self.w);
            x_global=self.x[idx_particle,...];
        
       
        self.bbox=np.array([x_global[0]-0.5*x_global[2], x_global[1]-0.5*x_global[3], x_global[2], x_global[3]]);
        ###########STEP 8: ACTUALIZACIÓN ADAPTATIVA DEL MODELO DE OBSERVACIÓN###########
        # Si la similitud combinada es alta, actualizamos el modelo de referencia
        #if final_similarity > cfg.update_threshold:
        #    # Actualizar histograma global de referencia
        #    self.hist_ref = (1 - cfg.learning_rate) * self.hist_ref + cfg.learning_rate * hist
        #    self.hist_ref = self.hist_ref / (self.hist_ref.sum() + 1e-10)
            
            # Actualizar el modelo espacial: calcular el nuevo conjunto de histogramas para la región candidata
        #    new_spatial = computeSpatialHSVHistograms(candidate_reg, self.K, grid=(2,2))
        #    for idx in range(n_cells):
        #        self.hist_ref_spatial[idx] = (1 - cfg.learning_rate) * self.hist_ref_spatial[idx] + cfg.learning_rate * new_spatial[idx]
        #        self.hist_ref_spatial[idx] = self.hist_ref_spatial[idx] / (self.hist_ref_spatial[idx].sum() + 1e-10)
