"""
This code learns to sub-sample MNIST images either in the image or Fourier domain and reconstructs or classifies them.
Copyright (C) 2020  Iris Huijben

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

For questions, you can contact us at: i.a.m.huijben@tue.nl
"""

"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : main_MNIST.py
                    This main file calls the model to sub-sample MNIST images either in the Fourier or image domain and reconstruct/classify them.
                    
    Author        : Iris Huijben
    Date          : 27/07/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019
==============================================================================
"""

import sys,  os.path as path, os
import numpy as np
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import myModels
import tensorflow as tf
from keras import backend as K
import dataLoader

#=============================================================================
reconVSclassif = 'recon'      #fill in 'recon'  or 'classif' to indicate image reconsturction or classification
classifAfterRecon = False     # Perform classification with input from a reconstruction network
database = 'MNIST'            #fill in 'MNIST' or 'CIFAR10'
domain = 'Fourier'            # fill in 'Fourier' or 'image' to indicate the sampling domain
#=============================================================================

if reconVSclassif == 'recon' and classifAfterRecon:
	print('ERROR: Detected ambiguity: do you want to perform classification or reconstruction?')

# Sub-sampling parameters
comp = 32                  	# Sub-sampling factor N/M
DPSsamp = True              # If true, sub-sampling is learned by DPS. If false, we use a fixed sub-sampling pattern (uniform or random)
gumbelTopK = True
Bahadir = False                  # Train sampling scheme according to Badair et al. (2019)
uniform = False             # In case DPSsamp is False, we use a non-trainable (fixed) sampling pattern which is either uniform, circular or random
circle = False              # In case DPSsamp is False, we use a non-trainable (fixed) sampling pattern which is either uniform, circular or random. Only implemented for comp=4 or comp=8

	
    
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

folds = 5                   # Amount of unfoldings in the reconstruction network
n_convs = 6                 # number of convolutional layers in the prox
share_prox_weights = False  # Boolean whether weights are shared over the unfoldings in the proximal mapping

# Training parameters
mux_out = int(np.ceil(((input_dim[0])*(input_dim[1]))//comp))     # Multiplexer output dims: the amount of samples to be sampled from the input
n_epochs = 500                      # Nr of epochs for training
    
    
batch_size =256                                                        # Batch size for training
batchPerEpoch = np.int32(np.ceil(x_train.shape[0]/batch_size))          # Number of batches used per epoch (of the training set)
learningrate = 2e-4                                                     # Learning rate of the reconstruction part of the network
subSampLrMult = 10                                       # Multiplier of learning rate for trainable unnormalized logits in A (with respect to the learning rate of the reconstruction part of the network)
tempIncr = 2                                           # Multiplier for temperature  parameter of softmax function. The temperature is kept at this constant value during trianing
        
# Parameters for entropy penalty multiplier (EM)                                                   
startEM = 0                 # Start value of the multiplier
endEM = 5e-4                # Value at endEpochEM-startEpochEM. Multiplier keeps increasing with the same linear rate per epoch
startEpochEM = 0            # Start epoch from which to start increasing the multiplier
endEpochEM = 50             # endEpochEM-startEpochEM is the time in which the multiplier increases from startEM to endEM
EM = [startEM, endEM, startEpochEM, endEpochEM]
entropyMult = K.variable(value=EM[0]) 
#%%
"""
=============================================================================
    Model definition
=============================================================================
"""

type_recon='automap_unfolded'

# create the generator
model = myModels.samplingTaskModel(
                            database,
                            domain,
                            input_dim, 
                            target_dim, 
                            comp, 
                            mux_out,
                            tempIncr,
                            entropyMult,
                            DPSsamp,
                            Bahadir,
                            uniform,
                            circle,
                            num_classes,
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

model.summary()


#%

"""
=============================================================================
    Initialize and compile model
=============================================================================
"""

## Define Optimizer:
import AdamWithLearningRateMultiplier
lr_mult = {}
lr_mult['CreateSampleMatrix'] = subSampLrMult   #for DPS and Gumbel top-K
lr_mult['prob_mask'] = subSampLrMult            #for LOUPE
optimizer = AdamWithLearningRateMultiplier.Adam_lr_mult(lr = learningrate, multipliers=lr_mult)

def SSIM(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=1)

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)
    
# Define loss and metric
if database == 'MNIST' and reconVSclassif == 'recon':
    loss = 'mean_squared_error'
    metrics = ['mean_squared_error', SSIM, PSNR]
    lossWeights = [1.]

else: #Classification
    loss = 'categorical_crossentropy' 
    metrics = ['accuracy']
    lossWeights = [1.]
    
# Compile model 
model.compile(optimizer=optimizer,loss=loss, loss_weights=lossWeights, metrics=metrics)


print("Learning rate: ", learningrate)
print("Temperature increase : ",  tempIncr)
print("subSampLrMult: ", subSampLrMult)
print("reconVSclassif: ", reconVSclassif)
print('Compression: ', comp)


#%
"""
=============================================================================
    Training
=============================================================================
"""

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
        
    
versionName = database+"_"+sampletype+"_"+reconVSclassif+"_{}_fact{}_lr{}_EM_{}-{}-{}-{}".format(domain,comp, learningrate, startEM, endEM, startEpochEM, endEpochEM)

if gumbelTopK:
    versionName = versionName+"_topK"

#savedirectory to save distribution at end of training. Make an empty list to not save anything
savedir = os.path.join(os.path.dirname(__file__),versionName)
if savedir:
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        
import trainingUpdate_callback
import trainableParams_callback
from keras.callbacks import LambdaCallback
        
def IncrEntropyMult(epoch):
    value = startEM+(endEM-startEM)/(endEpochEM-startEpochEM)*epoch
    print("EntropyPen mult:", value)
    K.set_value(entropyMult, value)
EntropyMult_cb = LambdaCallback(on_epoch_end=lambda epoch, log: IncrEntropyMult(epoch))



# Define callbacks
callbacks = [
        trainingUpdate_callback.training_callback(outputPerNepochs = 1, outputLastNepochs=(1,n_epochs), savedir=savedir, reconVSclassif=reconVSclassif),
        trainableParams_callback.weights_callback(outputPerNepochs = 1, outputLastNepochs = (1,n_epochs), mux_out = mux_out, tempIncr=tempIncr, domain = domain, DPSsamp=DPSsamp, topk=gumbelTopK, Bahadir=Bahadir, folds=folds, x_test=x_val,savedir=savedir,reconVSclassif = reconVSclassif),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, cooldown=20,verbose=1),
        ModelCheckpoint(os.path.join(savedir, 'weights-{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1),
        EntropyMult_cb
#        TensorBoard(log_dir=logdir,histogram_freq=20,write_grads=True)
        ]
"""
outputPerNEpochs:      How often do you want the callback to provide output
outputLastNepochs:     First argument indicates the X last epochs you want the callback to prpvide output
"""

History = model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs=n_epochs,
          validation_data=(x_val,y_val),
          callbacks= callbacks,
          verbose = 2)


#%%
"""
=============================================================================
    Save model
=============================================================================
"""

if savedir:
    model.save_weights(savedir+'\\'+versionName+'_weights.h5')
    print('Saved: ', versionName, ' in ' , savedir)
else:
    print("Not saved anything yet")
#

