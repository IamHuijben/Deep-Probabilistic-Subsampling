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
from keras.layers import Input, Flatten, Dense, LeakyReLU, Dropout, Reshape, Lambda, Conv2D, Add
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
        #print('keys: ', list(weights['prob_mask'].keys()))
        
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
        hardSamples[sampleCoord[0],sampleCoord[1]] = 1#sampleMask[sampleCoord[0],sampleCoord[1]]
        return [tf.constant(hardSamples), tf.constant(sampleMask)] #Output also the sampleMask for visualization
#    
    def compute_output_shape(self, input_shape):
        return [(self.input_dim[0],self.input_dim[1]),(self.input_dim[0],self.input_dim[1])]
    
    def get_config(self):
        base_config = super(CreateSampleMatrix, self).get_config()
        return base_config    


def InferenceModelLOUPE(comp,input_dim, database, domain, modelParams, num_classes,savedir,weightFile):
    

    ### Create subsampling network with hardsample mask ###
    input_layer = Input(shape=input_dim, name="Input")
    last_tensor = input_layer
    shape_input = input_layer.shape.as_list() #[BS, X,Y,channels]
    if input_dim[-1] == 1:
        last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
    

    Mask = CreateSampleMatrix(input_dim,comp, weightFile, savedir, name="HardSampleMask")(last_tensor)
    mask = Mask[0]
    Amatrix = mask
    #print('sampling mask: ', hardSample)
    
    upSampledInp = Lambda(lambda x: tf.multiply(x[0],tf.expand_dims(x[1],-1)), name="HardSampling")([last_tensor,mask])


    if modelParams[0] == 'classif':
        sensordata = Flatten()(upSampledInp)
    
        layer = Dense(shape_input[1]*shape_input[2],activation=None, name="Dense1")(sensordata)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(256,activation=None, name="Dense2")(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(128,activation=None, name="Dense3")(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(128,activation=None, name="Dense4")(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        classes = Dense(num_classes,activation='softmax', name="Softmax")(layer)
        
        model = Model(inputs = input_layer, outputs = classes)
    else: # reconVSclassif == 'recon0':
        share_prox_weights = modelParams[0]
        n_convs = modelParams[1]
        OneOverLmult = modelParams[2]
        folds = modelParams[3]
        import myModels
        

        def Subtract(inp):
            yZhat = inp[0]
            l = inp[1]
            return yZhat-l
        
        def Ax(inp):
            x = inp[0]
            Amatrix = inp[1]
            try:
                y = tf.multiply(x,Amatrix)
            except:
                y = tf.multiply(x,tf.expand_dims(Amatrix,-1))
            return y

        if share_prox_weights: 
            prox = []
            for n in range(0,n_convs):
                prox.append(Conv2D(64,3, activation='relu', padding='same'))
            prox.append(Conv2D(1,3, activation=None, padding='same'))

        # input:            
        Fx_k = upSampledInp
        
        if domain == 'Fourier':
            myModels.ifft2D(name="IFFT_meas")(Fx_k) 
        else:
            x = Fx_k
        
        if OneOverLmult:
            d = myModels.OneOverLmultiplier(name="StepSize_0")(x)
        else:
            d = Conv2D(1,3, activation=None, padding='same', name="StepSize_0")(x)
            
        
        x = d
        for k in range(0,folds):
        # iterate over the folds
        
            # apply proximal mapping
            if share_prox_weights:
                for n in range(0,n_convs):
                    x = prox[n](x)
                prox_out = prox[-1](x)   

            else:
                for n in range(0,n_convs):
                    layer_name = "conv2d_"+str(k*(n_convs+1)+n)
                    x = Conv2D(64,3, activation='relu', padding='same', name=layer_name)(x)
                prox_out = Conv2D(1,3, activation=None, padding='same', name="conv2d_"+str(k*(n_convs+1)+n_convs))(x)   
   
            # project back onto measurement space
            if domain == 'Fourier':
                x = myModels.fft2D(shape_in=shape_input,name="F_{}".format(k+1))(prox_out)
            else:
                x = prox_out
				
            x = Lambda(Ax)([x,Amatrix])  

            if domain == 'Fourier':			
                x = myModels.ifft2D(name="Ftran_{}".format(k+1))(x)


            if OneOverLmult:
                l = myModels.OneOverLmultiplier(name="StepSize_{}".format(k+1))(x)
            else:
                l = Conv2D(1,3, activation=None, padding='same',name="StepSize_{}".format(k+1))(x)
    
    
            # Subtract (1/L)*( Wd^T * Wd)) from I
            x = Lambda(Subtract)([prox_out,l])
            
            # Add B to output of S
            x = Add(name='Add_{}'.format(k+1))([x,d])

                  
        imagePred = prox_out   
        
        print(imagePred.shape)
        imagePred.set_shape((shape_input[0],shape_input[1],shape_input[2],1))            


        model = Model(inputs = input_layer, outputs = imagePred)
    

    # store weights before loading pre-trained weights
    preloaded_layers = model.layers.copy()
    
    preloaded_weights = []
    for pre in preloaded_layers:
        preloaded_weights.append(pre.get_weights())

    # load pre-trained weights
    model.load_weights(os.path.join(savedir,weightFile+'.hdf5'),by_name=True)

    # compare previews weights vs loaded weights
    for layer, pre in zip(model.layers, preloaded_weights):
        weights = layer.get_weights()
        
        if weights:
            if np.array_equal(weights, pre):
                print('not loaded', layer.name)
            else:
                print('loaded', layer.name)

    model.compile('sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model
    