"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : myModels.py
                    Model to subsample in Fourier domain and subsequently either reconstruct or classify the image
    Author        : Iris Huijben
    Date          : 24/07/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019

==============================================================================
"""
import sys, os
from pathsetupCIFAR10 import in_dir

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.models import load_model, Sequential
from keras.layers import GlobalAveragePooling2D, LeakyReLU,Dropout,Softmax, Flatten, Dense, Input, Conv2D, Reshape, Lambda, Add
from keras import regularizers
from keras import initializers
from keras import optimizers
import AdamWithLearningRateMultiplier

import temperatureUpdate
import tensorflow as tf
from keras.engine.topology import Layer
from Bahadir2019 import layers

# Define custom layers and regularizers
#######################################################################

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
                              regularizer = entropy_reg(self.entropyMult),
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
        base_config = super(CreateSampleMatrix_topK, self).get_config()
        return base_config    
      
#######################################################################    

def fixedHardUniformInit(shape,dtype=None):
    samplePattern = np.zeros((shape[1],shape[2],1))   
    comp = shape[0]
    
    if comp == 2:
        shift = comp*2
        reduceVerticalTrans= 1
        move = 0
    elif comp == 4:
        shift = comp*2
        reduceVerticalTrans = 2
        move = 0
    elif comp == 8:
        shift = comp 
        reduceVerticalTrans = 1
        move = 0
    elif comp == 16:
        shift = comp
        reduceVerticalTrans = 2
        move = 1
    elif comp == 28: #MNIST
        shift = comp//2
        reduceVerticalTrans = 1
        move = shift//8
    elif comp == 32: #CIFAR10 en MNIST. For MNIST this results in an actual compression factor of 31.36
        shift = comp//2
        reduceVerticalTrans= 1
        move = shift//8
    else:
        print('This factor is not yet implemented')
    
    firstRow = np.arange(move,samplePattern.shape[0],shift//2)
    secondRow = np.arange(shift//4+move,samplePattern.shape[0],shift//2)

    i,j = np.meshgrid(np.arange(move,samplePattern.shape[0],shift//(2*reduceVerticalTrans)), firstRow)
    samplePattern[i,j] = 1
    #
    k,l = np.meshgrid(np.arange(shift//(4*reduceVerticalTrans)+move,samplePattern.shape[0],shift//(2*reduceVerticalTrans)), secondRow)
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
    samplePattern = np.zeros((shape[1],shape[2],1))   
    comp = shape[0]
        
    if comp == 4:
        cr = 9
    if comp == 6:
        cr = 7.4
    elif comp == 8 and shape[1]==32: #CIFAR
        cr = 6.5
    elif comp == 8 and shape[1]==28: #MNIST
        cr = 5.6
    elif comp == 16 and shape[1]==96:
        cr = 13.55
    elif comp ==16 and shape[1] == 64:
        cr = 9       
    elif comp == 32: #Only for the MNIST case
        cr = 2.9
    else:
        print('No implementation yet available for this compression factor')

    # specify circle parameters: centre ij and radius
    ci,cj=shape[1]/2-0.5, shape[2]/2-0.5
    
    # Create index arrays to z
    I,J=np.meshgrid(np.arange(samplePattern.shape[0]),np.arange(samplePattern.shape[1]))
    
    # calculate distance of all points to centre
    dist=np.sqrt((I-ci)**2+(J-cj)**2)
    
    # Assign value of 0.2 to outer points
    samplePattern[np.where(dist<cr)]=1
    
    
    print(np.sum(samplePattern))
    print(shape[1]*shape[2]/16)
    
    
    # In some cases add some pixels to make sure the correct amount of pixels is used for a given subsampling factor
    if comp == 8 and shape[1] == 32: #Cifar case
        samplePattern[10,12] = 1
        samplePattern[10,19] = 1
        samplePattern[21,12] = 1
        samplePattern[21,19] = 1
    if comp == 6 and shape[1] == 32:
        samplePattern[12,9] = 0
        samplePattern[19,22] = 0
        
        
    if comp == 8 and shape[1] == 28: #MNIST case
        samplePattern[8,12] = 1
        samplePattern[8,15] = 1
        
    if comp == 32 and shape[1] == 28: #MNIST case
        samplePattern[11,12] = 1


    plt.imshow(samplePattern[:,:,0])

        
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
    # Where mux_in is the original amount of signal elements and mux_out is the amount of signal elements to be selected
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
        
class OneOverLmultiplier(Layer):
    def __init__(self, name, **kwargs):
        super(OneOverLmultiplier, self).__init__(name=name,**kwargs)
                 
    def build(self, input_shape):
        # Create a trainable multiplier
        self.multiplier = self.add_weight(name='OneOverL', 
                                      shape=(1,),
                                      initializer=initializers.Constant(value=1),
                                      trainable=True)

        super(OneOverLmultiplier, self).build(input_shape)

    def call(self, x):
        return self.multiplier * x
    
    def compute_output_shape(self, input_shape):
        return input_shape

######################################################################
#Own implementation of the 2D inverse Fourier transform, as the tf built-in function gave unexpected behaviour
class ifft2D(Layer):
    def __init__(self, name=None,**kwargs):
        super(ifft2D, self).__init__(name=name,**kwargs) 
    
    def build(self, input_shape):              
        super(ifft2D, self).build(input_shape)  
    
    def call(self, inp):   
        # inp is a Fourier transformed 2D image [BS,X,Y,2].
        # The function returns the inverse Fourier transformed image
        X = inp.shape.as_list()[2]
        Y = inp.shape.as_list()[1]
        
        complexInp = tf.dtypes.complex(inp[:,:,:,0],inp[:,:,:,1])
            
        # Define the 1D IDFT of size XxX
        j = np.arange(X)
        k = j
        jv,kv = np.meshgrid(j,k)
        DFT_x = np.exp((-2*np.pi*1j*jv*kv)/X)
        IDFT_x = (1/X)*DFT_x**(-1)
        
        # Define the 1D IDFT of size YxY
        j = np.arange(Y)
        k = j
        jv,kv = np.meshgrid(j,k)
        DFT_y = np.exp((-2*np.pi*1j*jv*kv)/Y)
        IDFT_y = (1/Y)*DFT_y**(-1)
           
        res1Dfourier = tf.einsum('db,abc->adc',tf.constant(IDFT_y,dtype=tf.complex64),complexInp)
        return tf.expand_dims(tf.abs(tf.einsum('cd,abc->abd', tf.constant(IDFT_x,dtype=tf.complex64), res1Dfourier)),-1)
        #[BS,X,Y,1]
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],1)
    
######################################################################  
#Own implementation of the 2D Fourier transform, as the tf built-in function gave unexpected behaviour
class fft2D(Layer):
    def __init__(self, shape_in=[], name=None,**kwargs):
        self.shape_in = shape_in
        super(fft2D, self).__init__(name = name, **kwargs) 
    
    def build(self, input_shape):  
        super(fft2D, self).build(input_shape)  
    
    def call(self, inp):   
        # inp is a 2D matrix that needs Fourier transformation
        # The function returns the Fourier transformed image

        X = self.shape_in[2]
        Y = self.shape_in[1]
                
        # Define the 1D IDFT of size XxX
        j = np.arange(X)
        k = j
        jv,kv = np.meshgrid(j,k)
        DFT_x = np.exp((-2*np.pi*1j*jv*kv)/X)
        
        # Define the 1D IDFT of size YxY
        j = np.arange(Y)
        k = j
        jv,kv = np.meshgrid(j,k)
        DFT_y = np.exp((-2*np.pi*1j*jv*kv)/Y)
           
        res1Dfourier = tf.einsum('db,abce->adce',tf.constant(DFT_y,dtype=tf.complex64),tf.dtypes.cast(inp,tf.complex64))
        res2Dfourier = tf.einsum('cd,abce->abde', tf.constant(DFT_x,dtype=tf.complex64), res1Dfourier) #[BS,X,Y,1] complex
        
        FourierReal = tf.squeeze(tf.real(res2Dfourier),-1)
        FourierImag = tf.squeeze(tf.imag(res2Dfourier),-1)
        return tf.stack([FourierReal, FourierImag], axis=-1)
        #[BS,X,Y,2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],2)

######################################################################

def samplingTaskModel(database, domain, input_dim, target_dim, comp, mux_out, tempIncr, entropyMult, DPSsamp, Bahadir, uniform, circle, num_classes, folds, reconVSclassif, n_epochs, batchPerEpoch, share_prox_weights = False, OneOverLmult=1, n_convs=3,learningrate=0.0001,type_recon=None,gumbelTopK=False):


    mux_in = input_dim[0]*input_dim[1]
    nrSelectedSamples = (mux_in)//comp
    
    input_layer = Input(shape=input_dim, name="Input")
    
    shape_input = input_layer.shape.as_list() #[BS, X,Y,2] 
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
            logits = CreateSampleMatrix(mux_out=mux_out,mux_in=mux_in, entropyMult=entropyMult, name="CreateSampleMatrix")(input_layer)              
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
            samples = topKsampling(mux_in, nrSelectedSamples, tempIncr, n_epochs, output_shape, batchPerEpoch,name="OneHotArgmax")([logits,input_layer])
            print('hard samples: ', samples.shape)    

        Amatrix = Lambda(Amat, name="AtranA_0")(samples)
        upSampledInp = Lambda(Ax, name="HardSampling")([input_layer,Amatrix])
        print('Shape after inverse A: ', upSampledInp.shape)

    elif Bahadir and comp >1: #Use method for learned sampling scheme by Bahadir et al (2019)
        
        # inputs
        last_tensor = input_layer
        # if necessary, concatenate with zeros for FFT
        if input_dim[-1] == 1:
            print('concat zero')
            last_tensor = layers.ConcatenateZero(name='concat_zero')(last_tensor)
            
        if domain == 'Fourier':
            # input -> kspace via FFT
            last_tensor = layers.FFT(name='fft')(last_tensor)

        # build probability mask
        prob_mask_tensor = layers.ProbMask(name='prob_mask', slope=5, initializer=None)(last_tensor) 
        
        prob_mask_tensor = layers.RescaleProbMap(sparsity=mux_out/mux_in, name='prob_mask_scaled')(prob_mask_tensor)

        # Realization of probability mask
        thresh_tensor = layers.RandomMask(name='random_mask')(prob_mask_tensor) 
        last_tensor_mask = layers.ThresholdRandomMask(slope=12, name='sampled_mask')([prob_mask_tensor, thresh_tensor]) 

        # Under-sample and back to image space via IFFT
        last_tensor = layers.UnderSample(name='HardSampling')([last_tensor, last_tensor_mask])
        upSampledInp = last_tensor
        print('upSampledInp shape: ', upSampledInp.shape)
#
        Amatrix = last_tensor_mask
        print(' shape Amatrix: ' , Amatrix.shape)
        
        
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


        def Ax(inp):
            x = inp[0]
            Amatrix = inp[1]
            y = tf.multiply(x,Amatrix)
            return y

        upSampledInp = Lambda(Ax)([input_layer,Amatrix])
        print('Shape after inverse A: ', upSampledInp.shape)
        
    # No sub-sampling                                             
    else:
        upSampledInp = input_layer
    

    if reconVSclassif == 'recon':

        if type_recon=='automap_unfolded': #Used for cifar10 reconstruction
            
            """
            =============================================================================
               RECONSTRUCTION NETWORK: unfolded scheme with learned proximal mapping
            =============================================================================
            """  
     
            def Subtract(inp):
                yZhat = inp[0]
                l = inp[1]
                return yZhat-l
    
            if share_prox_weights: 
                prox = []
                for n in range(0,n_convs):
                    prox.append(Conv2D(64,3, activation='relu', padding='same'))
                prox.append(Conv2D(1,3, activation=None, padding='same'))
    
            # input:            
            Fx_k = upSampledInp

            
            if domain == 'Fourier':
                x = ifft2D(name="IFFT_meas")(Fx_k) 
            else:
                x = Fx_k
            
            if OneOverLmult:
                d = OneOverLmultiplier(name="StepSize_0")(x)
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
                    startConv = 0 #Use another value here to match the names of a loaded model for inference
                    for n in range(0,n_convs):
                        layer_name = "conv2d_"+str(startConv + k*(n_convs+1)+n) #TODO: Remove the 4
                        x = Conv2D(64,3, activation='relu', padding='same', name=layer_name)(x)
                    prox_out = Conv2D(1,3, activation=None, padding='same', name="conv2d_"+str(startConv + k*(n_convs+1)+n_convs))(x)   
       
                # project back onto measurement space
                if domain == 'Fourier':
                    x = fft2D(shape_in=shape_input,name="F_{}".format(k+1))(prox_out)
                else:
                    x = prox_out

                x = Lambda(Ax)([x,Amatrix])   
                
                if domain == 'Fourier':
                    x = ifft2D(name="Ftran_{}".format(k+1))(x)
    
    
                if OneOverLmult:
                    l = OneOverLmultiplier(name="StepSize_{}".format(k+1))(x)
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
    
        else:
            print('No other models implemented')

    else: #Used for MNIST Classification 
    
        sensordata = Flatten()(upSampledInp)
      
        layer = Dense(shape_input[1]*shape_input[2],activation=None, name="Dense1")(sensordata)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(256,activation=None, name="Dense2")(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(128,activation=None, name="Dense3")(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(128,activation=None, name="Dense4")(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        classes = Dense(num_classes,activation='softmax', name="Softmax")(layer)
        
               
          
        model = Model(inputs = input_layer, outputs = classes)
            
    return model

   

def discriminator(in_shape=(48,48,1),learningrate=0.0002):
    
    input_layer = Input(shape=in_shape)
    layer = Conv2D(128, (3,3), strides=(2,2), padding='same')(input_layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(128, (3,3), strides=(2,2), padding='same')(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(128, (3,3), strides=(2,2), padding='same')(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = GlobalAveragePooling2D()(layer)
    layer = Dropout(0.4)(layer)
    out = Dense(1, activation='sigmoid')(layer)
    
    model = Model(inputs=input_layer,outputs = out)
	# compile model
    opt = optimizers.Adam(lr=learningrate, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model



def extract_layers(main_model, input_dim, starting_layer_ix, ending_layer_ix):
  # create an empty model
    input_layer = Input(shape=(input_dim[0],input_dim[1],1))
    layer  = input_layer
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
        layer = curr_layer(layer)
        # copy this layer over to the new model
    model = Model(inputs=input_layer, outputs=layer)   
        
    return model

def GAN(G, D, input_dim, subSampLrMult,learningrate=0.0002, loss_weights = [1,1,1]):

    D.trainable = False
    
    input_layer = Input(shape=input_dim)
    G_out = G(input_layer)
    D_out= D(G_out)
    
    #D_feat = extract_layers(D,input_dim,0,7) 
    #D_feat.summary()
    
    #D_feat.trainable = False 
    #D_feat_out = D_feat(G_out)
    D.summary()
    
    #model = Model(inputs=input_layer,outputs=[G_out,D_feat_out,D_out])
    model = Model(inputs=input_layer,outputs=[G_out,D_out])

    #Define Optimizer:
    lr_mult = {}
    lr_mult['CreateSampleMatrix'] = subSampLrMult #For DPS and Gumbel top-K
    lr_mult['prob_mask'] = subSampLrMult #For LOUPE
    optimizer = AdamWithLearningRateMultiplier.Adam_lr_mult(lr = learningrate, beta_1=0.5, multipliers=lr_mult)
    

    #model.compile(loss=['mean_squared_error','mean_squared_error','binary_crossentropy'], optimizer=optimizer, loss_weights=loss_weights)
    model.compile(loss=['mean_squared_error','binary_crossentropy'], optimizer=optimizer, loss_weights=loss_weights)
    
    
    return model 


# select real samples
def generate_real_samples(x_train,y_train, n_samples):
    # choose random instances
    ix = np.random.randint(0, x_train.shape[0], n_samples)
    # select images
    X = x_train[ix]
    X_target = y_train[ix]
    X_label = np.ones((n_samples, 1)) 
    
    return X, X_target, X_label
 
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, x_train, n_samples):
	# generate points in latent space
	# predict outputs
	X = generator.predict(x_train)
	# create class labels
	X_label = np.zeros((n_samples, 1))
	return X, X_label

# train the generator and discriminator
import matplotlib.pyplot as plt
def train(g_model, d_model, gan_model, x_train, y_train, x_val, y_val, EM, entropyMult, n_epochs=100, n_batch=128, savedir=[]):

    bat_per_epo = int(x_train.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    try:
        model_dist = Model(inputs = g_model.layers[0].output, outputs = g_model.get_layer("CreateSampleMatrix").output)
        model_samplingscheme =  Model(inputs = g_model.layers[0].output, outputs = g_model.get_layer("OneHotArgmax").output)
        learned_sampling = 1        
    except:
        learned_sampling = 0
    if learned_sampling==0:
        try:
            model_samplingscheme =  Model(inputs = g_model.layers[0].output, outputs = g_model.get_layer("sampled_mask").output)
            learned_sampling = 2        
        except:
            learned_sampling = 0
       
    
    input_shape = (y_train.shape[1],y_train.shape[2],1)
    #D_feat = extract_layers(d_model,input_shape,0,7)
    
    loss_store = np.zeros((n_epochs,3))
    mse_val = np.zeros((n_epochs))
    
    def IncrEntropyMult(epoch, EM):
        value = EM[0]+(EM[1]-EM[0])/(EM[3]-EM[2])*epoch
        print("EntropyPen mult:", value)
        K.set_value(entropyMult, value)
        
    
    # manually enumerate epochs
    for i in range(n_epochs):
        
        #Update the entropy penalty 
        IncrEntropyMult(i,EM)
              
        #numerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, X_real_target,X_real_label = generate_real_samples(x_train, y_train, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real_target, X_real_label)
 
    
           # generate 'fake' examples
            X_fake, X_fake_label = generate_fake_samples(g_model, X_real, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, X_fake_label)
 
    
            X_real, X_real_target,X_real_label = generate_real_samples(x_train, y_train, n_batch)
           # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            
            #X_feat_target = D_feat.predict(X_real_target)

            # update the generator via the discriminator's error
            #g_loss = gan_model.train_on_batch(X_real, [X_real_target,X_feat_target,y_gan])
            g_loss = gan_model.train_on_batch(X_real, [X_real_target,y_gan])
            
            
            
            if np.mod(j,100)==0:
                # summarize loss on this batch
                print('>{}, {}/{}, d1={}, d2={} g={}'.format(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

            
        loss_store[i,:] = g_loss

        # Inference on validation set:
        y_pred = g_model.predict(x_val)
        mse_val[i] = np.mean((y_pred-y_val)**2) 


        if learned_sampling == 1: #DPS
            # Display sample pattern DPS     
            shape = np.shape(X_real)
            
            logits = model_dist.predict_on_batch(tf.zeros((1,shape[1],shape[2],shape[3])))
            unnormDist = np.exp(logits)
            dist = np.transpose(np.transpose(unnormDist) / np.sum(unnormDist,1))
            
            samples = model_samplingscheme.predict_on_batch(tf.zeros((1,shape[1],shape[2],shape[3])))[0]
            pattern = np.reshape(np.sum(samples,axis=0),(shape[1],shape[2]))
            
            plt.figure(figsize=(8,8))
            plt.imshow(dist,cmap='hot_r')
            plt.colorbar(shrink=0.2)
            
            plt.figure(figsize=(4,4))
            plt.imshow(-pattern,cmap='gray')
            
        elif learned_sampling == 2: #LOUPE
            shape = np.shape(X_real)
            pattern = model_samplingscheme.predict_on_batch(tf.zeros((1,shape[1],shape[2],shape[3])))[0,:,:,0]
            
            plt.figure(figsize=(4,4))
            plt.imshow(-pattern,cmap='gray')
            
            
        plt.figure(figsize=(10,3))
        plt.subplot(131)
        plt.plot(loss_store[:i,0])
        plt.ylabel('Total train loss')
        plt.xlabel('Epochs')
        plt.subplot(132)
        plt.plot(loss_store[:i,1],label='train')
        plt.plot(mse_val[:i],label='val')
        plt.legend()
        plt.ylabel('Image MSE train loss')        
        plt.xlabel('Epochs')
#        plt.subplot(143)
#        plt.plot(loss_store[:i,2])
#        plt.ylabel('Feature MSE train loss')  
#        plt.xlabel('Epochs')
        plt.subplot(133)
        plt.plot(loss_store[:i,2])
        plt.ylabel('Discriminator train loss')         
        plt.xlabel('Epochs')
        
        
        # Display examples       
        plt.figure(figsize=(12,4))
        for k in range(0,5):
            plt.subplot(2,5,k+1)
            plt.imshow(y_val[k,:,:,0],cmap='gray')
            plt.subplot(2,5,6+k)
            plt.imshow(y_pred[k,:,:,0],cmap='gray')             
            
            
        plt.pause(.1)
        
        
        g_model.save_weights(os.path.join(savedir, 'weights-'+str(i)+'-val_loss-'+str(np.round(mse_val[i],5))+'.hdf5'))

    return g_model


