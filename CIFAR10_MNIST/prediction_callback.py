
"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : prediction_callback.py
                    Callback which displays the predicted and corresponding ground truth images
                    If a save directory is provided, the plots are automatically saved.
    Author        : Iris Huijben
    Date          : 30/07/2019
    Reference     : TODO
==============================================================================
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.models import Model
import tensorflow as tf

class output_callback(keras.callbacks.Callback):
     def __init__(self, outputPerNEpochs,outputLastNepochs, x_test, y_test,savedir, reconVSclassif):
         self.outputPerNepochs = outputPerNEpochs
         self.outputLastNepochs = outputLastNepochs[0]
         self.n_epochs = outputLastNepochs[1]
         self.x_test = x_test
         self.y_test = y_test
         self.savedir = savedir
         self.reconVSclassif = reconVSclassif
       

     def on_train_begin(self, logs={}):             
         return
 
     def on_train_end(self, logs={}):        
        return
 
     def on_epoch_begin(self, epoch, logs={}):
         return
         
     def on_epoch_end(self, epoch, logs={}):
        if self.reconVSclassif == 'recon':
            self.imageModel = Model(inputs = self.model.input, outputs=self.model.get_layer("ImageOutput").output)
            
            if (epoch+1) % self.outputPerNepochs == 0 or (epoch+1) > (self.n_epochs-self.outputLastNepochs):       
             
                predTest = self.imageModel.predict(self.x_test)[0]
       
                #Plot original and reconstruction image
                plt.figure()
                plt.imshow(self.y_test[0,:,:,0], cmap='gray',vmin=0,vmax=1)
                #plt.title('Original image')
                plt.axis('off')
    
                plt.figure()
                plt.imshow(predTest[:,:,0],cmap='gray',vmin=0,vmax=1)
                #plt.title('Reconstructed image')
                plt.axis('off')
                
                if self.savedir and (epoch+1) == self.n_epochs:
                    plt.savefig(self.savedir+'\Prediction.png',bbox_inches='tight')
                    plt.savefig(self.savedir+'\Prediction.svg',bbox_inches='tight')
                plt.pause(.1)

        return