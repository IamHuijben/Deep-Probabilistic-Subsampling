"""
This code learns to sub-sample and reconstruct synthetically generated images containing circles and lines.
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

    Source Name   : main.py
                    This main file calls the model to subsample and subsequently reconstruct synthetically generated images containing circles and lines.
                    
    Author        : Iris Huijben
    Date          : 09/08/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019
==============================================================================
"""

import os
import numpy as np
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from matplotlib import pyplot as plt
import tensorflow as tf
import myModel
from keras import backend as K

#=============================================================================
reconVSclassif = 'recon'    #fill in 'recon'  or 'classif' to indicate image reconstruction or classification
#=============================================================================


"""
=============================================================================
    Create dataset
=============================================================================
"""

num_samples = int(5e4)
batch_size = 128                                # Batch size for data generators

input_size = (32,32)
num_classes = 2

cases = np.random.randint(0, num_classes, num_samples)

# Generate images that contain a circle and a number of horizontal lines
def datagenmix(batch_size): 
    
    #Settings for circles
    I,J=np.meshgrid(np.arange(input_size[0]),np.arange(input_size[1]))  # Create a mesh
    cr_max = 4
    cr_min = 2
    amp_max = 10
    amp_min = 1
    maxlines = 5
    
    while True:
        
        x = np.zeros((batch_size, input_size[0], input_size[1],1))

        for i in range(0,batch_size):
            
            cr = cr_min+np.random.rand()*(cr_max-cr_min)                             #radius
        
            ci,cj= np.random.randint(cr,32-np.ceil(cr)), np.random.randint(cr,np.ceil(32-cr)) #Center points
            dist=np.sqrt((I-ci)**2+(J-cj)**2)                                   # Calculate distance of all points to centre
            innerArea = np.where(dist<cr)
        
            x[i,innerArea[0],innerArea[1]] =  amp_min+np.random.rand()*(amp_max-amp_min)
    
            # horizontal line
            nrlines = np.random.randint(0,maxlines)
            bound = 3
            
            for j in range(0,nrlines):
                ypos = np.random.randint(bound,32-bound)
                x[i,ypos,:,0] = amp_min+np.random.rand()*(amp_max-amp_min)
            
        if reconVSclassif == 'recon':
            xF = np.empty((batch_size, input_size[0], input_size[1],2))
            xFcomplex = np.fft.fftshift(np.fft.fft2(x,axes=(1,2))[:,:,:,0],axes=(1,2))
            xF[:,:,:,0] = np.real(xFcomplex)
            xF[:,:,:,1] = np.imag(xFcomplex)
    
            y = x
            yield(xF,y)
        else:
            print('classification is not implemented for this case')

            
gen = datagenmix(batch_size)
x_test, y_test = next(gen)

#%%
"""
=============================================================================
    Parameter definitions
=============================================================================
"""
# Subsampling parameters
comp = 32               # Subsampling factor N/M
DPSsamp = True        # If true, subsampling is learned by DPS. If false, we use a fixed subsampling pattern 
gumbelTopK = False      # Boolean indicating whether to use top-k sampling if DPSsamp = True (vs top-1 sampling)
Bahadir= False          # Boolean indicating whether to use the trainable sampling method (called LOUPE) proposed by Bahadir et al (2019)
uniform = False         # Boolean indicating whether to use a fixed uniform sampling scheme
circle = False          # Boolean indicating whether to use a fixed circular sampling scheme

learningrate = 1e-4

# Parameters for entropy penalty multiplier (EM)                                                   
startEM = 2e-4                 # Start value of the multiplier
endEM = 2e-4                # Value at endEpochEM-startEpochEM. Multiplier keeps increasing with the same linear rate per epoch
startEpochEM = 0            # Start epoch from which to start increasing the multiplier
endEpochEM = 50             # endEpochEM-startEpochEM is the time in which the multiplier increases from startEM to endEM
EM = [startEM, endEM, startEpochEM, endEpochEM]
entropyMult = K.variable(value=EM[0]) 


if DPSsamp ==True:
    sampletype = 'DPS'
else:
    if gumbelTopK:
        sampletype = 'GumbelTopK'
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
        

input_dim = x_test.shape[1:]                   # Dimensions of the inputs to the network: [fourier bins, IQ components]
target_dim = y_test.shape[1:]                  # Dimension of the targets for the network: [time steps]

mux_out = np.prod(input_dim[0:2])//comp       # Multiplexer output dims: the amount of samples to be sampled from the input
tempIncr = 5                                  # Multiplier for temperature  parameter of softmax function. The temperature drops from (tempIncr*TempUpdateBasisTemp) to (tempIncr*TempUpdateFinalTemp) defined in temperatureUpdate.py file
if reconVSclassif == 'recon':
    subSampLrMult = 10                        # Multiplier of learning rate for trainable unnormalized logits in A (with respect to the learning rate of the reconstruction part of the network)
else:
    subSampLrMult = 50

versionName = "Toy_"+sampletype+"_{}_fact{}_lr{}_lrMult{}_EM_{}-{}-{}-{}".format(reconVSclassif,comp, learningrate, subSampLrMult,startEM, endEM, startEpochEM, endEpochEM)

# Training parameters
n_epochs = 4000                                # Number of epochs during training
batchPerEpoch = num_samples//batch_size        # Number of batches used per epoch


"""
=============================================================================
    Model definition
=============================================================================
"""

model = myModel.full_model(
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
                    n_epochs, 
                    batchPerEpoch, 
                    gumbelTopK)

## Print model summary:
model.summary()

#%%

"""
=============================================================================
    Initialize and compile model
=============================================================================
"""

## Define Optimizer:
import AdamWithLearningRateMultiplier
lr_mult = {}
lr_mult['CreateSampleMatrix'] = subSampLrMult
lr_mult['prob_mask'] = subSampLrMult
optimizer = AdamWithLearningRateMultiplier.Adam_lr_mult(lr = learningrate, multipliers=lr_mult)

# Define loss and metric
if reconVSclassif == 'recon':
    def SSIM(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=1)

    def PSNR(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1)
    
    loss = 'mean_squared_error' 
    metrics = [SSIM, PSNR,'mean_squared_error']
else: #Classification
    loss = 'categorical_crossentropy' 
    metrics = ['accuracy']


# Compile model 
model.compile(optimizer=optimizer,loss= loss,metrics=metrics)

print("Learning rate: ", learningrate)
print("Temperature increase : ",  tempIncr)
print("subSampLrMult: ", subSampLrMult)
print('Compression: ', comp)

#%%
"""
=============================================================================
    Training
=============================================================================
"""
import trainableParams_callback
import trainingUpdate_callback
import prediction_callback  
from keras.callbacks import LambdaCallback

#save directory to save distribution at end of training. Make an empty list to not save anything
savedir = os.path.join(os.path.dirname(__file__),versionName)
if savedir:
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        
def IncrEntropyMult(epoch):
    value = startEM+(endEM-startEM)/(endEpochEM-startEpochEM)*epoch
    print("EntropyPen mult:", value)
    K.set_value(entropyMult, value)
    
EntropyMult_cb = LambdaCallback(on_epoch_end=lambda epoch, log: IncrEntropyMult(epoch))



callbacks = [ 
        trainableParams_callback.weights_callback(outputPerNepochs=100, outputLastNepochs = (1,n_epochs), mux_out = mux_out, tempIncr=tempIncr,DPSsamp=DPSsamp,Bahadir=Bahadir, x_test=x_test,savedir=savedir),
        trainingUpdate_callback.training_callback(outputPerNepochs = 100, outputLastNepochs=(1,n_epochs), reconVSclassif=reconVSclassif, savedir=savedir),
        prediction_callback.output_callback(outputPerNEpochs = 100, outputLastNepochs=(1,n_epochs),x_test = x_test, y_test=y_test,reconVSclassif=reconVSclassif,savedir=savedir),
        ModelCheckpoint(os.path.join(savedir, 'weights-{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, cooldown=20,verbose=1),
        EntropyMult_cb
#        TensorBoard(log_dir=logdir,histogram_freq=20,write_grads=True)
        ]
"""
outputPerNEpochs:      How often do you want the callback to provide output
outputLastNepochs:     First argument indicates the X last epochs you want the callback to prpvide output
"""

model.fit_generator(gen, 
                    steps_per_epoch=batchPerEpoch, 
                    epochs = n_epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data = (x_test, y_test))

"""
=============================================================================
    Save model weights
=============================================================================
"""

if savedir:
    model.save_weights(savedir+'\\'+versionName+'_weights.h5')
    print('Saved: ', versionName, ' in ' , savedir)
else:
    print("Not saved anything yet")
#