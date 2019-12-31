"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : trainableParams_callbacks.py
                    Callback which displays at the end of each X epochs:
                        * The trained distributions
                        * Soft-samples from these distributions and the corresponding temperature parameter
                        * Hard samples from these distributions
                    If a save directory is provided, the plots are automatically saved.
                    
    Author        : Iris Huijben
    Date          : 26/04/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing"
==============================================================================
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
import temperatureUpdate
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from matplotlib import cm
from matplotlib.colors import ListedColormap


class weights_callback(keras.callbacks.Callback):
    def __init__(self, outputPerNepochs, outputLastNepochs, mux_out, tempIncr,DPSsamp, Bahadir, x_test,savedir):
        self.outputPerNepochs = outputPerNepochs
        self.mux_out = mux_out
        self.epoch = 0
        self.outputLastNepochs = outputLastNepochs[0]
        self.n_epochs = outputLastNepochs[1]
        self.tempIncr = tempIncr
        self.DPSsamp = DPSsamp
        self.Bahadir = Bahadir
        self.x_test = x_test
        self.shape = self.x_test.shape
        self.savedir = savedir
        self.fontsize = 14
        self.inputChan = self.x_test.shape[-1]
        
        ### Create custom colormap that is suitable for plotting the trained distributions
        cmap = cm.get_cmap('gist_heat_r',256)
        newcolors = cmap(np.linspace(0, 1, 512))[50:]
        self.SparseBasesMap = ListedColormap(newcolors)
        newcolors = cmap(np.linspace(0, 1, 256))[10:]
        white = np.array([1,1,1,1])
        newcolors[0, :] = white #Make sure the lowest values are white
        self.myCmap = ListedColormap(newcolors)
              
        
    def on_train_begin(self, logs={}):

 
        if self.mux_out < 1024:
            if not self.Bahadir:
                self.modelHidden =  Model(inputs = self.model.input, outputs = self.model.get_layer("CreateSampleMatrix").output)
                dist = self.modelHidden.predict_on_batch(tf.zeros((1,32,32,self.inputChan)))
                
        
                # Plot the initial distributions, either trained (if self.DPSsamp is true) or fixed
                if self.DPSsamp:
                    #self.modelHidden1 =  Model(inputs = self.model.input, outputs = self.model.get_layer("SoftSampling").output)
                    self.modelHidden2 =  Model(inputs = self.model.input, outputs = self.model.get_layer("OneHotArgmax").output)
                    self.AtranA = Model(inputs =self.model.input, outputs = self.model.get_layer("AtranA_0").output)
                    
                    #Plot the initial weight distribution
                    unnormDist = np.exp(dist)
                    normDist = np.transpose(np.transpose(unnormDist) / np.sum(unnormDist,1))
                    
                    plt.figure()
                    plt.gcf().clear()
                    plt.imshow(normDist,cmap=self.myCmap,interpolation='nearest', aspect='equal')
                    plt.xlabel('Initial pixels', fontsize=14)
                    plt.ylabel('Selected pixels', fontsize=14)
                    plt.colorbar()
                    plt.title('Initial distributions on Fourier coeffiecents',fontsize=14)
                    plt.pause(.1)
                    
                else:
                    plt.figure()
                    plt.gcf().clear()
                    plt.imshow(dist[:,:,0],cmap=self.myCmap,interpolation='nearest', aspect='equal')
                    plt.xticks(fontsize=self.fontsize)
                    plt.yticks(fontsize=self.fontsize)
                    plt.savefig(self.savedir+'\hardSamplesWithoutLabels.png',bbox_inches='tight')
                    plt.savefig(self.savedir+'\hardSamplesWithoutLabels.svg',bbox_inches='tight')
                    #plt.pause(.1)
    
                    plt.xlabel('X', fontsize=14)
                    plt.ylabel('Y', fontsize=14)
                    #plt.title('Sampled Fourier coefficients',fontsize=14)
                    
                    plt.savefig(self.savedir+'\hardSamples.png',bbox_inches='tight')
                    plt.savefig(self.savedir+'\hardSamples.svg',bbox_inches='tight')
                    plt.pause(.1)
                    
                                       
            if self.Bahadir: #Bahadir method
                self.SamplingMask =  Model(inputs = self.model.input, outputs = self.model.get_layer("sampled_mask").output)

                        
            
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return


    def on_epoch_end(self, epoch, logs={}):
        
        aspect = 'auto'
        if (epoch+1) % self.outputPerNepochs == 0 or (epoch+1) > (self.n_epochs-self.outputLastNepochs) :       

            print('====================================================')
            
            if self.mux_out < 1024 and not self.Bahadir:

                Temp = temperatureUpdate.temperature_update_numeric(self.tempIncr, epoch, self.n_epochs)
                print('Temperature: ', round(Temp,2))
      
                logits = self.modelHidden.predict_on_batch(tf.zeros((1,32,32,self.inputChan)))
                
                if self.DPSsamp:
                    unnormDist = np.exp(logits)
                    normDist = np.transpose(np.transpose(unnormDist) / np.sum(unnormDist,1))
    
                       
                    plt.figure()
                    plt.imshow(normDist,cmap=self.myCmap,interpolation='nearest',aspect=aspect)
                    plt.xlabel('Initial pixels', fontsize=14)
                    plt.ylabel('Selected pixels', fontsize=14)
                    plt.colorbar()
                    plt.title('Trained distributions on pixels',fontsize=14)
                
                    if self.savedir and (epoch+1) == self.n_epochs:
                        plt.savefig(self.savedir+'\distributions.png',bbox_inches='tight')
                        plt.savefig(self.savedir+'\distributions.svg',bbox_inches='tight')
                    plt.pause(.1) 
                      
#                    softSamples = self.modelHidden1.predict_on_batch(tf.zeros((1,32,32,self.inputChan)))
#                    softSample = softSamples[0,:,:] #Plot the first of the batch only
#                    plt.figure()
#                    plt.imshow(softSample,cmap=self.myCmap,interpolation='nearest',aspect=aspect)
#                    plt.xlabel('Initial pixels', fontsize=14)
#                    plt.ylabel('Selected pixels', fontsize=11)
#                    plt.colorbar()
#                    plt.title('Soft samples',fontsize=14)
#                    
#                    if self.savedir and (epoch+1) == self.n_epochs:
#                        plt.savefig(self.savedir+'\softSamples.png',bbox_inches='tight')
#                        plt.savefig(self.savedir+'\softSamples.svg',bbox_inches='tight')
#                    plt.pause(.1) 
#                                 
                    #Plot the hard samples
                    hardSample = self.AtranA.predict_on_batch(tf.zeros((1,32,32,self.inputChan)))[0,:,:,0]
    
                    plt.figure()
                    plt.imshow(hardSample,cmap=self.myCmap,interpolation='nearest',aspect='equal')
                    plt.xticks(fontsize=self.fontsize)
                    plt.yticks(fontsize=self.fontsize)
                    plt.xlabel('X', fontsize=self.fontsize)
                    plt.ylabel('Y', fontsize=self.fontsize)
                    #plt.title('Sampled Fourier coefficients',fontsize=14)
                
                    if self.savedir and (epoch+1) == self.n_epochs:
                        plt.savefig(self.savedir+'\hardSamples.png',bbox_inches='tight')
                        plt.savefig(self.savedir+'\hardSamples.svg',bbox_inches='tight')
                        plt.pause(.1)
                        
                        #Also save without label
                        plt.figure()
                        plt.imshow(hardSample,cmap=self.myCmap,interpolation='nearest',aspect='equal')
                        plt.xticks(fontsize=self.fontsize)
                        plt.yticks(fontsize=self.fontsize)
                        plt.savefig(self.savedir+'\hardSamplesWithoutLabels.png',bbox_inches='tight')
                        plt.savefig(self.savedir+'\hardSamplesWithoutLabels.svg',bbox_inches='tight')
                    plt.pause(.1) 
                    
        elif self.mux_out < (self.x_test.shape[1]*self.x_test.shape[2]) and self.Bahadir:

                hardSample = self.SamplingMask.predict_on_batch(tf.zeros((1,self.shape[1],self.shape[2],self.shape[3])))[0]
                
                print("shape samples: " , hardSample.shape)
                print('Nr of selected samples: ', np.sum(hardSample))

                plt.figure()
                plt.imshow(hardSample[...,0],cmap=self.myCmap,interpolation='nearest',aspect='equal')

                plt.xlabel('X', fontsize=self.fontsize)
                plt.ylabel('Y', fontsize=self.fontsize)
                plt.colorbar()
                plt.xticks(fontsize=self.fontsize)
                plt.yticks(fontsize=self.fontsize)
                #plt.title('Sampled Fourier coefficients',fontsize=14)
            
                if self.savedir and (epoch+1) == self.n_epochs:
                    plt.savefig(self.savedir+'\hardSamples.png',bbox_inches='tight')
                    plt.savefig(self.savedir+'\hardSamples.svg',bbox_inches='tight')
                    plt.pause(.1)
                    
                    #Also save a version without labels
                    plt.figure()
                    plt.imshow(np.fft.fftshift(hardSample[...,0]),cmap=self.myCmap,interpolation='nearest',aspect='equal')
                    plt.xticks(fontsize=self.fontsize)
                    plt.yticks(fontsize=self.fontsize)             
                    plt.savefig(self.savedir+'\hardSamplesWithoutLabels.png',bbox_inches='tight')
                    plt.savefig(self.savedir+'\hardSamplesWithoutLabels.svg',bbox_inches='tight')
                    
                plt.pause(.1) 
                    
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
