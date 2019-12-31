"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : trainingUpdate_callback.py
                    Callback which displays the training graph for the train 
                    and validation set at the end of each X epochs. 
                    If a save directory is provided, the graph is saved

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
     def __init__(self, outputPerNepochs, outputLastNepochs,savedir,reconVSclassif):
         self.outputPerNepochs = outputPerNepochs
         self.outputLastNepochs = outputLastNepochs[0]
         self.n_epochs = outputLastNepochs[1]
         self.savedir = savedir
         self.reconVSclassif = reconVSclassif
         self.train_MSE_im = []
         self.val_MSE_im = []
         self.train_PSNR_im = []
         self.val_PSNR_im = []
         self.train_SSIM_im = []
         self.val_SSIM_im = []
         self.train_MSE_feat = []
         self.val_MSE_feat = []
         self.train_acc = []
         self.val_acc = []
         
     def on_train_begin(self, logs={}):        
         return
 
     def on_train_end(self, logs={}):
         return
 
     def on_epoch_begin(self, epoch, logs={}):
         return
         
     def on_epoch_end(self, epoch, logs={}):
    
        if self.reconVSclassif == 'recon':
            self.train_MSE_im.append(logs.get('ImageOutput_mean_squared_error'))
            self.val_MSE_im.append(logs.get('val_ImageOutput_mean_squared_error'))
            self.train_PSNR_im.append(logs.get('ImageOutput_PSNR'))
            self.val_PSNR_im.append(logs.get('val_ImageOutput_PSNR'))
            self.train_SSIM_im.append(logs.get('ImageOutput_SSIM'))
            self.val_SSIM_im.append(logs.get('val_ImageOutput_SSIM'))
            self.train_MSE_feat.append(logs.get('FeatureOutput_mean_squared_error'))
            self.val_MSE_feat.append(logs.get('val_FeatureOutput_mean_squared_error'))
        else:
            self.train_acc.append(logs.get('acc'))
            self.val_acc.append(logs.get('val_acc'))
			
        if (epoch+1) % self.outputPerNepochs == 0 or (epoch+1) > (self.n_epochs-self.outputLastNepochs):       
             
            if self.reconVSclassif == 'recon':
                plt.figure(figsize=(10,10))
                plt.gcf().clear()   
                plt.subplot(221)
                plt.plot(np.arange(epoch+1),self.train_MSE_im)
                plt.plot(np.arange(epoch+1),self.val_MSE_im)
                plt.title('MSE - images')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.legend(['Train','Val'], loc='upper right')
                plt.grid()
				
                plt.subplot(222)
                plt.plot(np.arange(epoch+1),self.train_PSNR_im)
                plt.plot(np.arange(epoch+1),self.val_PSNR_im)
                plt.title('PSNR - images')
                plt.xlabel('Epoch')
                plt.ylabel('PSNR')
                plt.legend(['Train','Val'], loc='lower right')
                plt.grid()
                				
                plt.subplot(223)
                plt.plot(np.arange(epoch+1),self.train_SSIM_im)
                plt.plot(np.arange(epoch+1),self.val_SSIM_im)
                plt.title('SSIM - images')
                plt.xlabel('Epoch')
                plt.ylabel('SSIM')
                plt.legend(['Train','Val'], loc='lower right')
                plt.grid()
                				
                plt.subplot(224)
                plt.plot(np.arange(epoch+1),self.train_MSE_feat)
                plt.plot(np.arange(epoch+1),self.val_MSE_feat)
                plt.title('MSE - features')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.legend(['Train','Val'], loc='upper right')
                plt.grid()
            else:
                plt.figure()
                plt.plot(np.arange(epoch+1),self.train_acc)
                plt.plot(np.arange(epoch+1),self.val_acc)
                plt.title('Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Acc')
                plt.legend(['Train','Val'], loc='lower right')
                plt.grid()

            if self.savedir:                 
                plt.savefig(self.savedir+'\\TrainingGraph.svg',bbox_inches='tight')
                plt.savefig(self.savedir+'\\TrainingGraph.png',bbox_inches='tight')
            plt.pause(.1)
        
        
        return