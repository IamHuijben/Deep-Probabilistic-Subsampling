"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : main_CIFAR10.py
                    This main file calls the model to su-sample CIFAR10 images in the Fourier domain and reconstruct/classify them.
                    
    Author        : Iris Huijben
    Date          : 27/07/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019

==============================================================================
"""

import sys,  os.path as path, os
import numpy as np
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import myModels
import tensorflow as tf
import cv2 
from keras import backend as K
import dataLoader


#=============================================================================
reconVSclassif = 'recon'    #fill in 'recon'  or 'classif' to indicate image reconsturction or classification
database = 'CIFAR10'            #fill in 'MNIST' or 'CIFAR10'
domain = 'image'          # fill in 'Fourier' or 'image' to indicate the sampling domain
#=============================================================================

    
"""
=============================================================================
    Load and prepare the datasets
=============================================================================
"""

[x_train, y_train, x_val, y_val, x_test, y_test] = dataLoader.LoadData(database, domain, reconVSclassif)
input_dim = x_train.shape[1:]
target_dim = y_train.shape[1:]
num_classes = target_dim[-1]

#%%

"""
=============================================================================
    Parameter definitions
=============================================================================
"""

# Sub-sampling parameters
comp = 8                    # Sub-sampling factor N/M
DPSsamp = True            # If true, sub-sampling is learned by DPS. If false, we use a fixed sub-sampling pattern (uniform or random)
gumbelTopK = False
Bahadir = False
uniform = False             # In case DPSsamp is False, we use a non-trainable (fixed) sampling pattern which is either uniform, circular or random
circle = False              # In case DPSsamp is False, we use a non-trainable (fixed) sampling pattern which is either uniform, circular or random. Only implemented for comp=4 or comp=8

folds = 5                   # Amount of unfoldings for the reconstruction part: LISTA
n_convs = 6                 # number of convolutional layers in the prox
share_prox_weights = False  # If true, one proximal operator is trained for all unfoldings. If false, every unfolding trains a separate proximal operator.

# Training parameters
mux_out = int(np.ceil(((input_dim[0])*(input_dim[1]))//comp))     # Multiplexer output dims: the amount of samples to be sampled from the input
n_epochs = 100                                      # Number of epochs for training

batch_size = 8                                                        # Batch size for training
batchPerEpoch = np.int32(np.ceil(x_train.shape[0]/batch_size))          # Number of batches used per epoch (of the training set)
learningrate = 0.0001                                # Learning rate of the reconstruction part of the network
subSampLrMult = 5                                     # Multiplier of learning rate for trainable unnormalized logits in A (with respect to the learning rate of the reconstruction part of the network)  
tempIncr = 2                                          # Multiplier for temperature  parameter of softmax function. The temperature is kept at this constant value during trianing
                              
# Parameters for entropy penalty multiplier (EM)                             
startEM = 1e-6              # Start value of the multiplier
endEM = 1e-6                # Value at endEpochEM-startEpochEM. Multiplier keeps increasing with the same linear rate per epoch
startEpochEM = 0            # Start epoch from which to start increasing the multiplier
endEpochEM = 100            # endEpochEM-startEpochEM is the time in which the multiplier increases from startEM to endEM
EM = [startEM, endEM, startEpochEM, endEpochEM]
entropyMult = K.variable(value=EM[0])  
#%%
"""
=============================================================================
    Model definition
=============================================================================
"""

discriminator = myModels.discriminator(in_shape=target_dim,learningrate=learningrate)

type_recon='automap_unfolded'
D_weight= 0.004             #Weight of the discriminator loss
loss_weights = [1,D_weight]
#loss_weights = [1,0,D_weight] #Generator output, Discriminator feature output, Disriminator classification output

# create the generator
generator = myModels.samplingTaskModel(
                            database,
                            [],
                            input_dim, 
                            target_dim, 
                            comp, 
                            mux_out,
                            tempIncr,
                            entropyMult,
                            subSampLrMult,
                            DPSsamp,
                            Bahadir,
                            uniform,
                            circle,
                            [],
                            folds,
                            reconVSclassif,
                            n_epochs,
                            batchPerEpoch,
                            share_prox_weights = share_prox_weights,
                            OneOverLmult = 0, # 0 means train a convolutional kernel instead
                            n_convs = n_convs,
                            learningrate=learningrate,
                            type_recon=type_recon,
                            gumbelTopK=gumbelTopK)

generator.summary()


# create the gan
gan_model = myModels.GAN(generator, 
                           discriminator, 
                           input_dim, 
                           subSampLrMult,
                           learningrate=learningrate,
                           loss_weights = loss_weights) #% mse vs gan

## Print model summary:
gan_model.summary()

#%%
"""
=============================================================================
    Create save directory
=============================================================================
"""
# In case DPSsamp is False, we use a non-trainable (fixed) sampling pattern which is either uniform, circular or random. Only implemented for comp=4 or comp=8

if DPSsamp ==True:
    sampletype = 'DPS'
else:
    if Bahadir == True:
        sampletype = 'loupe'
    elif circle == True:
        sampletype = 'lpf'
    elif uniform == True:
        sampletype = 'uniform'
    elif uniform == False and circle == False:
        sampletype = 'random'
    else:
        sampletype = 'unknown'
        
versionName = database+"_"+sampletype+"_proxgrad_folds{}_convs{}_fact{}_lrMult{}_lr{}_EM_{}-{}-{}-{}_patience10_cooldown20".format(folds,n_convs,comp,subSampLrMult, learningrate, startEM, endEM, startEpochEM, endEpochEM)

if gumbelTopK:
    versionName = versionName+"_topK"
    
#savedirectory to save distribution at end of training. Make an empty list to not save anything
savedir = os.path.join(os.path.dirname(__file__),versionName)

#%%
"""
=============================================================================
    Train
=============================================================================
"""

# train model
generator = myModels.train(generator, 
               discriminator, 
               gan_model, 
               x_train, 
               y_train, 
               x_val, 
               y_val, 
               EM,
               entropyMult,
               n_epochs=n_epochs, 
               n_batch=batch_size,
               savedir = savedir) 

#%%
"""
=============================================================================
    Save model
=============================================================================
"""
#%%
if savedir:
    generator.save_weights(savedir+'\\''weights.h5')
    print('Saved: ', versionName, ' in ' , savedir)
else:
    print('Not saved anything yet')

