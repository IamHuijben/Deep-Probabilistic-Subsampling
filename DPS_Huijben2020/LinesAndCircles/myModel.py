"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : myModel.py
                    Model to subsample over pixels and subsequently reconstruct the image
    Author        : Iris Huijben
    Date          : 09/08/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019

==============================================================================
"""
import sys, os
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, Conv2D , Dropout, MaxPooling2D
from keras.layers import LeakyReLU, Activation, Reshape, Lambda, Add, Softmax, Flatten
from Bahadir2019 import layers

from keras import regularizers
from keras import initializers

import temperatureUpdate

import tensorflow as tf
from scipy.optimize import fsolve
from keras.engine.topology import Layer
  

class entropy_reg(tf.keras.regularizers.Regularizer):

    def __init__(self, entropyMult):
        self.entropyMult = entropyMult

    def __call__(self, logits):
        normDist = tf.nn.softmax(logits,1)
        logNormDist = tf.log(normDist+1e-20)
        
        rowEntropies = -tf.reduce_sum(tf.multiply(normDist, logNormDist),1)
        sumRowEntropies = tf.reduce_sum(rowEntropies)
        
        multiplier = self.entropyMult
        return multiplier*sumRowEntropies

    def get_config(self):
        return {'entropyMult': float(self.entropyMult)}

#######################################################################

# Create the trainable logits matrix
class CreateSampleMatrix(Layer):
    def __init__(self,mux_in,mux_out,entropyMult,name=None,**kwargs):
        self.mux_in = mux_in
        self.mux_out = mux_out
        self.entropyMult = entropyMult
        super(CreateSampleMatrix, self).__init__(name=name,**kwargs)

    def build(self, input_shape):
                       
        self.kernel = self.add_weight(name='TrainableLogits', 
                              shape=(self.mux_out, self.mux_in),
                              initializer = initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
                              regularizer=entropy_reg(self.entropyMult),
                              trainable=True)
        super(CreateSampleMatrix, self).build(input_shape)  

    def call(self, x):    
        return self.kernel
    
    def compute_output_shape(self, input_shape):
        return (self.mux_out, self.mux_in)
    
    def get_config(self):
        base_config = super(CreateSampleMatrix, self).get_config()
        return base_config    
    
#######################################################################    
    
# Create the trainable logits matrix
class CreateSampleMatrix_topK(Layer):
    def __init__(self,mux_in,mux_out,name=None,**kwargs):
        self.mux_in = mux_in
        self.mux_out = mux_out
        super(CreateSampleMatrix_topK, self).__init__(name=name,**kwargs)

    def build(self, input_shape):
                       
        self.kernel = self.add_weight(name='TrainableLogits', 
                              shape=(self.mux_out, self.mux_in),
                              initializer = initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
                              trainable=True)
        super(CreateSampleMatrix_topK, self).build(input_shape)  

    def call(self, x):    
        return self.kernel
    
    def compute_output_shape(self, input_shape):
        return (self.mux_out, self.mux_in)
    
    def get_config(self):
        base_config = super(CreateSampleMatrix, self).get_config()
        return base_config    
    
######################################################################    
    
def fixedHardUniformInit(shape,dtype=None):
    samplePattern = np.zeros((shape[1],shape[2],1))   
    comp = shape[0]
    

    if comp == 2:
        shift = comp*2
        reduceVerticalTrans= 1
        moveHor = 0
        moveVer = moveHor
    elif comp == 4:
        shift = comp*2
        reduceVerticalTrans = 2
        moveHor = 0
        moveVer = moveHor
    elif comp == 8:
        shift = comp 
        reduceVerticalTrans = 1
        moveHor = 0
        moveVer = moveHor
    elif comp == 16:
        shift = comp
        reduceVerticalTrans = 2
        moveHor = 1
        moveVer = moveHor
    elif comp == 32:
        shift = comp//2
        reduceVerticalTrans= 1
        moveHor = shift//8
        moveVer = moveHor
    elif comp == 64:
        shift = comp//2
        reduceVerticalTrans = 2
        moveHor = 3 
        moveVer = 2
    elif comp == 256:
        samplePattern[7,23] = 1
        samplePattern[7,7] = 1
        samplePattern[23,23] = 1
        samplePattern[23,7] = 1
        return samplePattern

    
    firstRow = np.arange(moveHor,samplePattern.shape[0],shift//2)
    secondRow = np.arange(shift//4+moveHor,samplePattern.shape[0],shift//2)
    
    i,j = np.meshgrid(np.arange(moveVer,samplePattern.shape[0],shift//(2*reduceVerticalTrans)), firstRow)
    samplePattern[i,j] = 1
    #
    k,l = np.meshgrid(np.arange(shift//(4*reduceVerticalTrans)+moveVer,samplePattern.shape[0],shift//(2*reduceVerticalTrans)), secondRow)
    samplePattern[k,l] = 1
        
    return samplePattern

# Create non-trainable uniform sub-sampling matrix
class CreateFixedUniformSampleMatrix(Layer):
    def __init__(self,comp,name=None,**kwargs):
        self.comp = comp
        super(CreateFixedUniformSampleMatrix, self).__init__(name=name,**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='Weights', 
                              shape=(self.comp,input_shape[1],input_shape[2]),
                              initializer = fixedHardUniformInit,
                              trainable=False)
        super(CreateFixedUniformSampleMatrix, self).build(input_shape)  
    def call(self, x):   
        return self.kernel        
    
    def compute_output_shape(self, input_shape):
        return (input_shape[1],input_shape[2],1)
    
    def get_config(self):
        base_config = super(CreateFixedUniformSampleMatrix, self).get_config()
        return base_config    


######################################################################
def fixedHardCircularInit(shape,dtype=None):
    comp = shape[0]
    samplePattern = np.zeros((shape[1],shape[2],1))   
        
    if comp == 4:
        cr = 9
    elif comp == 8:
        cr = 6.5
    elif comp == 16:
        cr = 4.5
    elif comp == 32:
        cr = 3
    else:
        print('No implementation yet available for this compression factor')
  

    # specify circle parameters: centre ij and radius
    ci,cj=15.5,15.5
    
    # Create index arrays to z
    I,J=np.meshgrid(np.arange(samplePattern.shape[0]),np.arange(samplePattern.shape[1]))
    
    # calculate distance of all points to centre
    dist=np.sqrt((I-ci)**2+(J-cj)**2)
    
    # Assign value 1 to pixels within the circle
    samplePattern[np.where(dist<cr)]=1
    
    # In some cases, add some extra 1s to make sure the exact same amount of samples is taken for circular (fixed) sampling, and DPS with this compression factor
    if comp == 8:
        samplePattern[10,12] = 1
        samplePattern[10,19] = 1
        samplePattern[21,12] = 1
        samplePattern[21,19] = 1
        
    if comp == 16:
        samplePattern[12,12] = 1
        samplePattern[12,19] = 1
        samplePattern[19,19] = 1
        samplePattern[19,12] = 1
    
 
    return samplePattern

# Create non-trainable circular sub-sampling matrix
class CreateFixedCircularSampleMatrix(Layer):
    def __init__(self,comp,name=None,**kwargs):
        self.comp = comp
        super(CreateFixedCircularSampleMatrix, self).__init__(name=name,**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='Weights', 
                              shape=(self.comp,input_shape[1],input_shape[2]),
                              initializer = fixedHardCircularInit,
                              trainable=False)
        super(CreateFixedCircularSampleMatrix, self).build(input_shape)  
    def call(self, x):   
        return self.kernel        
    
    def compute_output_shape(self, input_shape):
        return (input_shape[1],input_shape[2],1)
    
    def get_config(self):
        base_config = super(CreateFixedCircularSampleMatrix, self).get_config()
        return base_config   
######################################################################

def fixedHardRandomInit(shape,dtype=None):    
    #Take random samples
    npInd = np.random.permutation(np.arange(shape[1]*shape[2]))[0:shape[0]]
    ind = tf.constant(npInd, dtype = tf.int32)
    return tf.reshape(tf.reduce_sum(tf.one_hot(ind,shape[1]*shape[2]),axis=0),(shape[1],shape[2],1))

# Create non-trainable random sub-sampling matrix
class CreateFixedRandomSampleMatrix(Layer):
    def __init__(self,mux_out,name=None,**kwargs):
        self.mux_out = mux_out
        super(CreateFixedRandomSampleMatrix, self).__init__(name=name,**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='Weights', 
                              shape=(self.mux_out,input_shape[1],input_shape[2]),
                              initializer = fixedHardRandomInit,
                              trainable=False)
        super(CreateFixedRandomSampleMatrix, self).build(input_shape)  
    def call(self, x):   
        return self.kernel        
    
    def compute_output_shape(self, input_shape):
        return (input_shape[1],input_shape[2],1)
    
    def get_config(self):
        base_config = super(CreateFixedRandomSampleMatrix, self).get_config()
        return base_config 

######################################################################    
 
# Create a mask for the logits dependent on the already sampled classes. 
# This mask ensures that one class is only sampled once over the M distributions in the next layer
def MaskingLogits(inp):
    logits = inp[0]
    inpData = inp[1]
    mux_in = logits.shape[1]
    mux_out = logits.shape[0]
    mux_out_scalar = logits.shape.as_list()[0]
     
    # Create gumbel noise, which is different for every patch in the batch. So size of GN: [Batch size, mux_out, mux_in]
    # Where mux_in is the original amount of Fourier bins of the signal and mux_out is the amount to be selected
    GN = -tf.log(-tf.log(tf.random_uniform(tf.stack([tf.shape(inpData)[0],mux_out,mux_in],0),0,1)+1e-20)+1e-20) 
    
    #Repeat the logits over (dynamic) batch size by adding zeros of this shape
    dummyForRepOverBS = tf.zeros_like(GN)
    logitsRep = logits + dummyForRepOverBS
        
    mask = tf.ones(tf.stack([tf.shape(inpData)[0],mux_in],0))
    maskedLogits = [None]*mux_out_scalar
      
    #Shuffle rows in order to apply sampling without replacement in random row order  
    shuffledRows = np.arange(mux_out_scalar)
    np.random.shuffle(shuffledRows)
    
    unnormProbs = tf.exp(logitsRep)
    for i in range(mux_out_scalar):      
        row = shuffledRows[i]
        unnormProbRow = unnormProbs[:,row,:]
  
        maskedLogitRow = tf.log(tf.multiply(unnormProbRow,mask) + 1e-20)
        maskedLogits[row] = maskedLogitRow
    
        #Find next mask: change a one to a zero where the hard sample will be taken
        hardSampleForMasking = tf.one_hot(tf.argmax(maskedLogitRow+GN[:,row,:],axis=1),depth=mux_in)
        mask = mask - hardSampleForMasking
    maskedLogits = tf.stack(maskedLogits,axis=1)
    
    # Return GN as well to make sure the same GN is used in the softSampling layer
    return [maskedLogits, GN]

def MaskingLogits_output_shape(input_shape):
    outputShapeTup = (input_shape[1][0],)+input_shape[0]             
    return [outputShapeTup,outputShapeTup]

######################################################################    

# Apply row-based sampling from the Gumbel-softmax distribution, with a variable temperature parameter, depending on the epoch
class SoftSampling(Layer):
    def __init__(self,tempIncr=1,n_epochs=1, batchPerEpoch=32, name=None,**kwargs):
        self.tempIncr = tempIncr
        self.n_epochs = n_epochs
        self.batchPerEpoch = batchPerEpoch
        super(SoftSampling, self).__init__(name=name,**kwargs) 
    
    def build(self, input_shape):
        self.step = K.variable(0)
        super(SoftSampling, self).build(input_shape)  
    
    def call(self, inp):
        maskedLogits = inp[0]
        GN = inp[1]

        # Find temperature for gumbel softmax based on epoch and update epoch
        epoch = self.step/self.batchPerEpoch
        Temp = temperatureUpdate.temperature_update_tf(self.tempIncr, epoch, self.n_epochs)
        
        updateSteps = []
        updateSteps.append((self.step, self.step+1))
        self.add_update(updateSteps,inp)

        return tf.nn.softmax((maskedLogits+GN) / Temp, axis=-1)
            
    def compute_output_shape(self, input_shape):
        return (input_shape[0])

######################################################################  
        
# Apply hard sampling of the soft-samples (only in the forward pass)
# This hard sampling does happen unordered: non-sequential
def hardSampling(maskedSoftSamples):
    hardSamples = tf.one_hot(tf.argmax(maskedSoftSamples,axis=-1),depth=maskedSoftSamples.shape[-1])
    return tf.stop_gradient(hardSamples - maskedSoftSamples) + maskedSoftSamples

def identity_output_shape(input_shape):
    return input_shape

######################################################################    
# Layer that implements hard sampling in the forward pass, and soft sampling in the backwards pass using gumbel-softmax relaxation    
class topKsampling(Layer):
    
    def __init__(self,mux_in, mux_out, tempIncr, n_epochs, output_shape, batchPerEpoch, name=None,**kwargs):
        self.mux_in = mux_in
        self.mux_out =mux_out
        self.tempIncr = tempIncr
        self.n_epochs = n_epochs
        self.outShape = output_shape
        self.batchPerEpoch = batchPerEpoch
        super(topKsampling, self).__init__(name=name,**kwargs) 

    def build(self, input_shape):
        self.step = K.variable(0)
        super(topKsampling, self).build(input_shape)  
      
    def call(self, inp):
        logits = inp[0]
        inpData = inp[1]
        self.BS = tf.shape(inpData)[0]
        #logits shape is [1,mux_out]
       
        ### Forwards ###

        GN = -tf.log(-tf.log(tf.random_uniform((self.BS,1,self.mux_in),0,1)+1e-20)+1e-20) 
        perturbedLog = logits+GN
        
        topk = tf.squeeze(tf.nn.top_k(perturbedLog, k=self.mux_out)[1],axis=1) #[BS,k]      
        hardSamples = tf.one_hot(topk,depth=self.mux_in) #[BS,k,N]
        
        ### Backwards ###
        epoch = self.step/self.batchPerEpoch
        Temp = temperatureUpdate.temperature_update_tf(self.tempIncr, epoch, self.n_epochs)
        updateSteps = []
        updateSteps.append((self.step, self.step+1))
        self.add_update(updateSteps,inp)

        prob_exp = tf.tile(tf.expand_dims(tf.exp(logits),0),(self.BS,self.mux_out,1)) #[BS,K,N]        
        cumMask = tf.cumsum(hardSamples,axis=-2, exclusive=True) #[BS,K,N]
        
        softSamples = tf.nn.softmax((tf.log(tf.multiply(prob_exp,1-cumMask+1e-20))+tf.tile(GN,(1,self.mux_out,1)))/Temp, axis=-1)
        
        return tf.stop_gradient(hardSamples - softSamples) + softSamples
    
    def compute_output_shape(self, input_shape):
        return self.outShape



######################################################################

    
def full_model(input_dim, target_dim, comp, mux_out, tempIncr, entropyMult, DPSsamp, Bahadir, uniform, circle, n_epochs, batchPerEpoch, gumbelTopK):

    mux_in = input_dim[0]*input_dim[1]
    nrSelectedSamples = (mux_in)//comp
 
    input_layer = Input(shape=input_dim, name="Input")
    shape_input = input_layer.shape.as_list() #[BS, 32,32]
    print('shape input: ',shape_input)

    """
    =============================================================================
        SUB-SAMPLING NETWORK
    =============================================================================
    """
    if (DPSsamp or Bahadir) and comp > 1:

        def Amat(samples):
            Amatrix = tf.reshape(tf.reduce_sum(samples,axis=1),(-1,shape_input[1],shape_input[1],1))
            return Amatrix
     
        def Ax(inp):
            x = inp[0]
            Amatrix = inp[1]
            y = tf.multiply(x,Amatrix)
            return y
         
    if DPSsamp == True and comp > 1:
             
    
        mux_in = shape_input[1]*shape_input[2]


        if gumbelTopK == False:        
            logits = CreateSampleMatrix(mux_out=mux_out,mux_in=mux_in,entropyMult=entropyMult, name="CreateSampleMatrix")(input_layer)#(input_layerR)               
            print('logits shape: ', logits.shape)   
            
            maskedLogits = Lambda(MaskingLogits,name="MaskingLogits",output_shape=MaskingLogits_output_shape)([logits,input_layer])
            print('masked logits shape: ', maskedLogits[0].shape)   
          
            samples = SoftSampling(tempIncr=tempIncr,n_epochs=n_epochs,batchPerEpoch=batchPerEpoch,name="SoftSampling")(maskedLogits)
            print('soft samples shape: ', samples.shape)
            
            samples = Lambda(hardSampling,name="OneHotArgmax",output_shape=identity_output_shape)(samples)
            print('hard samples shape:' , samples.shape)
        else:
            logits = CreateSampleMatrix_topK(mux_out=1,mux_in=mux_in, name="CreateSampleMatrix")(input_layer)              
            print('logits shape: ', logits.shape)   
            
            output_shape = (shape_input[0],nrSelectedSamples,mux_in)
            samples = topKsampling(mux_in, nrSelectedSamples, tempIncr, n_epochs, output_shape, batchPerEpoch, name="OneHotArgmax")([logits,input_layer])
            print('hard samples: ', samples.shape)    
        
        Amatrix = Lambda(Amat, name="AtranA_0")(samples)
        upSampledInp = Lambda(Ax, name="HardSampling")([input_layer,Amatrix])
        print('Shape after inverse A: ', upSampledInp.shape)
 
    elif Bahadir and comp >1: #Use method for learned sampling scheme by Bahadir et al (2019)
        
        # inputs
        last_tensor = input_layer

        # build probability mask
        prob_mask_tensor = layers.ProbMask(name='prob_mask', slope=5, initializer=None)(last_tensor) 
        
        #Todo: is mux_out the right sparsity parameter here?
        prob_mask_tensor = layers.RescaleProbMap(sparsity=mux_out/mux_in, name='prob_mask_scaled')(prob_mask_tensor)

        # Realization of probability mask
        thresh_tensor = layers.RandomMask(name='random_mask')(prob_mask_tensor) 
        last_tensor_mask = layers.ThresholdRandomMask(slope=12, name='sampled_mask')([prob_mask_tensor, thresh_tensor]) 

        # Under-sample and back to image space via IFFT
        last_tensor = layers.UnderSample(name='HardSampling')([last_tensor, last_tensor_mask])
        upSampledInp = last_tensor
        print('upSampledInp shape: ', upSampledInp.shape)
        
    #Use a fixed sample scheme    
    elif comp > 1:
        if uniform:
            Amatrix = CreateFixedUniformSampleMatrix(comp=comp, name="CreateSampleMatrix")(input_layer)
            print('shape Amatrix: ' , Amatrix.shape)
            
        elif circle:
            Amatrix = CreateFixedCircularSampleMatrix(comp=comp, name="CreateSampleMatrix")(input_layer)
            print('shape Amatrix: ' , Amatrix.shape)
                      
        else: #Use a random fixed scheme
            Amatrix = CreateFixedRandomSampleMatrix(mux_out=mux_out, name="CreateSampleMatrix")(input_layer)
            print('shape Amatrix: ' , Amatrix.shape)

        upSampledInp = Lambda(lambda dummy: tf.multiply(Amatrix,input_layer),name="Hardsampling")([input_layer,Amatrix])
        print('Shape after inverse A: ', upSampledInp.shape)#    
    # No sub-sampling                                             
    else:
        upSampledInp = input_layer
    


    sensordata = Flatten()(upSampledInp)
    dense1 = Dense(shape_input[1]*shape_input[2],activation='tanh', name="dense_1")(sensordata)
    dense2 = Dense(shape_input[1]*shape_input[2],activation='tanh', name="dense_2")(dense1)
    dense_re = Reshape((shape_input[1],shape_input[2],1))(dense2)

    conv1 = Conv2D(64,5, activation='relu', padding='same', name="conv2d_1")(dense_re)
    conv2 = Conv2D(64,5, activation='relu', padding='same', name="conv2d_2")(conv1)
    output = Conv2D(1,7, activation=None, padding='same', name="conv2d_3")(conv2)


    return Model(inputs = input_layer, outputs = output)
    




