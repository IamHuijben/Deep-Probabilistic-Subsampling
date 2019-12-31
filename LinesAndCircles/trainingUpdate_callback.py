"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : trainingUpdate_callback.py
                    Callback which displays the training graph for the train 
                    and validation set at the end of each X epochs. 
                    If a save directory is provided, the graph is saved at the end of training.

    Author        : Iris Huijben
    Date          : 15/01/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019

==============================================================================
"""

import keras
import numpy as np
import matplotlib.pyplot as plt

class training_callback(keras.callbacks.Callback):
     def __init__(self, outputPerNepochs, outputLastNepochs,reconVSclassif,savedir):
         self.outputPerNepochs = outputPerNepochs
         self.outputLastNepochs = outputLastNepochs[0]
         self.n_epochs = outputLastNepochs[1]
         self.reconVSclassif = reconVSclassif
         self.savedir = savedir
         self.train_PSNR = []
         self.val_PSNR = []
         self.train_SSIM = []
         self.val_SSIM = []
         self.train_accuracy = []
         self.val_accuracy = []

     def on_train_begin(self, logs={}):        
         return
 
     def on_train_end(self, logs={}):
         return
 
     def on_epoch_begin(self, epoch, logs={}):
         return
         
     def on_epoch_end(self, epoch, logs={}):
    
        if self.reconVSclassif == 'recon':
            self.train_PSNR.append(logs.get('PSNR'))
            self.val_PSNR.append(logs.get('val_PSNR'))
            self.train_SSIM.append(logs.get('SSIM'))
            self.val_SSIM.append(logs.get('val_SSIM'))
        else:
            self.train_accuracy.append(logs.get('acc'))
            self.val_accuracy.append(logs.get('val_acc'))

 
        if (epoch+1) % self.outputPerNepochs == 0 or (epoch+1) > (self.n_epochs-self.outputLastNepochs):       
        
             if self.reconVSclassif == 'recon':
                 plt.figure(figsize=(10,4))
                 plt.subplot(121)
                 plt.plot(np.arange(epoch+1),self.train_PSNR)
                 plt.plot(np.arange(epoch+1),self.val_PSNR)
                 plt.title('Training and validation PSNR')
                 plt.xlabel('Epoch')
                 plt.ylabel('PSNR')
                 plt.legend(['Train','Val'], loc='lower right')
                 plt.grid()
                 
                 plt.subplot(122)
                 plt.plot(np.arange(epoch+1),self.train_SSIM)
                 plt.plot(np.arange(epoch+1),self.val_SSIM)
                 plt.title('Training and validation SSIM')
                 plt.xlabel('Epoch')
                 plt.ylabel('SSIM')
                 plt.legend(['Train','Val'], loc='lower right')
                 plt.grid()
             else:
                                 
                 plt.figure()
                 plt.plot(np.arange(epoch+1),self.train_accuracy)
                 plt.plot(np.arange(epoch+1),self.val_accuracy)
                 plt.title('Training and validation accuracy')
                 plt.xlabel('Epoch')
                 plt.ylabel('Accuracy')
                 plt.legend(['Train','Val'], loc='lower right')
                 plt.grid()
              
             if self.savedir and (epoch+1) == self.n_epochs:                 
                 plt.savefig(self.savedir+'\\TrainingGraph.svg',bbox_inches='tight')
                 plt.savefig(self.savedir+'\\TrainingGraph.png',bbox_inches='tight')
             plt.pause(.1)

        
        return