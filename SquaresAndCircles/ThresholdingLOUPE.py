"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : ThresholdedLOUPE.py
                    This main file load weights of a pretrained model using LOUPE (Bahadir et al,2019)
                    
    Author        : Iris Huijben
    Date          : 14/11/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019
==============================================================================
"""
# Code that thresholds the trained mask by LOUPE (Bahadir et al,2019) to induce the correct subsampling factor for inference

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from matplotlib import cm
from matplotlib.colors import ListedColormap
import h5py
from Bahadir2019 import layers
from keras.layers import Input, Flatten, Dense, Reshape, Conv2D, Lambda
from keras.engine.topology import Layer
import os


def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Create the trainable logits matrix
class CreateSampleMatrix(Layer):
    def __init__(self,input_dim,comp,weightFile, savedir,name=None,**kwargs):
        self.input_dim = input_dim
        self.comp = comp
        self.mux_out = (self.input_dim[0]*self.input_dim[1])//comp
        self.sparsity = self.mux_out/(self.input_dim[0]*self.input_dim[1])
        self.weightFile = weightFile
        self.savedir = savedir
        super(CreateSampleMatrix, self).__init__(name=name,**kwargs)

    def build(self, input_shape):
        super(CreateSampleMatrix, self).build(input_shape)  

    def call(self, x):    
    
        full_name = os.path.join(self.savedir,self.weightFile+'.hdf5')
        weights = h5py.File(full_name,'r+')    
        print('keys: ', list(weights['prob_mask'].keys()))
        #print('keys: ', list(weights.keys()))
        
        prob_mask_tensor = weights['prob_mask']['prob_mask']['logit_weights:0'][...,0] #The trained weights of the ProbMask layer
        
        #output of ProbMask layer
        prob_mask = sigmoid(5 * prob_mask_tensor) 
        
        #Output of RescaleProbMask layer
        xbar = np.mean(prob_mask)        
        r = self.sparsity / xbar        
        beta = (1-self.sparsity) / (1-xbar)

        renormMask = (prob_mask * r) if r <= 1 else (1 - (1 - prob_mask) * beta)
    
        # Output of thresholdRandomMask
        thresh = np.random.uniform(0.0,1.0,(self.input_dim[0],self.input_dim[1]))
        sampleMask = sigmoid(12 * (renormMask-thresh))

        
        # Make sure to only select M hard samples
        sampleCoord = largest_indices(sampleMask,self.mux_out)
        hardSamples = np.zeros_like(prob_mask)
        hardSamples[sampleCoord[0],sampleCoord[1]] = 1
        #hardSamples[sampleCoord[0],sampleCoord[1]] = sampleMask[sampleCoord[0],sampleCoord[1]] #Use the same value as used during training
        return [tf.constant(hardSamples), tf.constant(renormMask)] #Output also the sampleMask for visualization
#    
    def compute_output_shape(self, input_shape):
        return [(self.input_dim[0],self.input_dim[1]),(self.input_dim[0],self.input_dim[1])]
    
    def get_config(self):
        base_config = super(CreateSampleMatrix, self).get_config()
        return base_config    


def LoadModelLOUPE(comp,input_dim,savedir,weightFile):
        
    ### Create subsampling network with hardsample mask ###
    input_layer = Input(shape=input_dim, name="Input")
    last_tensor = input_layer
    shape_input = input_layer.shape.as_list() #[BS, X,Y,channels]

    # Create thresholded sample mask
    Mask = CreateSampleMatrix(input_dim,comp, weightFile, savedir, name="HardSampleMask")(input_layer)
  
    # Under-sample and back to image space via IFFT   
    upSampledInp = Lambda(lambda out: tf.multiply(last_tensor,tf.expand_dims(Mask[0],-1)), name="HardSampling")([last_tensor,Mask[0]])
    #print('hardsample shape: ' , upSampledInp.shape)

    
    sensordata = Flatten()(upSampledInp)
    dense1 = Dense(shape_input[1]*shape_input[2],activation='tanh')(sensordata)
    dense2 = Dense(shape_input[1]*shape_input[2],activation='tanh')(dense1)
    dense_re = Reshape((shape_input[1],shape_input[2],1))(dense2)

    conv1 = Conv2D(64,5, activation='relu', padding='same')(dense_re)
    conv2 = Conv2D(64,5, activation='relu', padding='same')(conv1)
    output = Conv2D(1,7, activation=None, padding='same')(conv2)

    model = Model(inputs = input_layer, outputs = output)
    model.load_weights(os.path.join(savedir,weightFile+'.hdf5'), by_name=True)


    return model

    