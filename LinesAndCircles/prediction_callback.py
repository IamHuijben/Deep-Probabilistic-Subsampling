
"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : prediction_callback.py
                    Callback which displays the predicted and corresponding ground truth images
                    If a save directory is provided, the plots are automatically saved.
    Author        : Iris Huijben
    Date          : 30/07/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019

==============================================================================
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

class output_callback(keras.callbacks.Callback):
     def __init__(self, outputPerNEpochs,outputLastNepochs, x_test, y_test,reconVSclassif, savedir):
         self.outputPerNepochs = outputPerNEpochs
         self.outputLastNepochs = outputLastNepochs[0]
         self.n_epochs = outputLastNepochs[1]
         self.x_test = x_test
         self.y_test = y_test
         self.reconVSclassif = reconVSclassif
         self.savedir = savedir
         self.fontsize= 16
         
         cmap = cm.get_cmap('Blues',256)
         newcolors = cmap(np.linspace(0, 1, 256))[:-30]
         self.cmap = ListedColormap(newcolors)
                   
         
     def on_train_begin(self, logs={}):        
        return
 
     def on_train_end(self, logs={}):        
        return
 
     def on_epoch_begin(self, epoch, logs={}):
         return
         
     def on_epoch_end(self, epoch, logs={}):
        
        
        if (epoch+1) % self.outputPerNepochs == 0 or (epoch+1) > (self.n_epochs-self.outputLastNepochs):       
            pred = self.model.predict_on_batch(self.x_test)


            if self.reconVSclassif == 'recon':
                #Plot the first two original and reconstructed images from the on-line generated test set
                origIm1 = self.y_test[0,:,:,0]
                origIm2 = self.y_test[1,:,:,0]
                predIm1 = pred[0,:,:,0]
                predIm2 = pred[1,:,:,0]
          
                plt.figure(figsize=(3,6))
                plt.subplot(121)
                plt.imshow(np.sqrt(self.x_test[0,:,:,0]**2+self.x_test[0,:,:,1]**2), cmap='jet')
                plt.axis('off')
             
                plt.subplot(122)
                plt.imshow(origIm1, cmap='gray',vmin=0,vmax=np.max(origIm1))
                plt.axis('off')
                #plt.title('Original image')
                
                #plt.subplot(223)
                plt.figure()
                plt.imshow(predIm1, cmap='gray',vmin=0,vmax=np.max(origIm1))
                plt.colorbar()
                plt.axis('off')
                #plt.title('Reconstructed image')
                
                if self.savedir and (epoch+1) == self.n_epochs:
                    plt.savefig(self.savedir+'\Prediction1.png',bbox_inches='tight')
                    plt.savefig(self.savedir+'\Prediction1.svg',bbox_inches='tight')
                plt.pause(.1)
                
                
                plt.figure(figsize=(3,6))
                plt.subplot(121)
                plt.imshow(np.sqrt(self.x_test[1,:,:,0]**2+self.x_test[1,:,:,1]**2), cmap='jet')
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(origIm2, cmap='gray',vmin=0,vmax=np.max(origIm2))
                plt.axis('off')
                #plt.title('Original image')
                                    
                #plt.subplot(224)
                plt.figure()
                plt.imshow(predIm2, cmap='gray',vmin=0,vmax=np.max(origIm2))
                plt.colorbar()
                plt.axis('off')
                #plt.title('Reconstructed image')
                
                if self.savedir and (epoch+1) == self.n_epochs:
                    plt.savefig(self.savedir+'\Prediction2.png',bbox_inches='tight')
                    plt.savefig(self.savedir+'\Prediction2.svg',bbox_inches='tight')
                plt.pause(.1)
                
            
            else: #Classification
                predictions = np.argmax(pred,axis=1)
                confusionMatrix = confusion_matrix(np.argmax(self.y_test,axis=1), predictions)
                plt.imshow(confusionMatrix, cmap = self.cmap)
                plt.title('Confusion matrix',fontsize=self.fontsize)
                plt.xlabel('Predicted class',fontsize=self.fontsize)
                plt.ylabel('Ground truth class',fontsize=self.fontsize)
                center = 0
                plt.text(0-center,1-center,confusionMatrix[1,0], color='black',fontsize=self.fontsize)
                plt.text(0+center,0+center,confusionMatrix[0,0], color='black',fontsize=self.fontsize)
                plt.text(1+center,0+center,confusionMatrix[0,1], color='black',fontsize=self.fontsize)
                plt.text(1-center,1+center,confusionMatrix[1,1], color='black',fontsize=self.fontsize)
                plt.xticks([0,1])
                plt.yticks([0,1])
                        
                plt.figure()
                plt.scatter(pred[:,0],pred[:,1])
                plt.title('Prediction values',fontsize=self.fontsize)
                plt.xlabel('Probability for class 0',fontsize=self.fontsize)
                plt.ylabel('Probability for class 1', fontsize=self.fontsize)
                plt.pause(.1)
        return