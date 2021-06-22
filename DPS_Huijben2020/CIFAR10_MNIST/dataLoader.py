"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : dataLoader.py
                    This file loads the data needed for training or inference scripts
                    
    Author        : Iris Huijben
    Date          : 18/11/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun
                    "Deep Probabilistic subsampling for Task-adaptive  Compressed Sensing", 2019
==============================================================================
"""

import os
import numpy as np

def LoadData(database, domain, reconVSclassif):
    
    if database == 'CIFAR10':
        from pathsetupCIFAR10 import in_dir
    else:
        from pathsetupMNIST import in_dir
    

    #Load presaved train, validation and test set   
    x_train = np.load(os.path.join(in_dir,"x_train.npy"))
    x_val = np.load(os.path.join(in_dir,"x_val.npy"))
    x_test = np.load(os.path.join(in_dir,"x_test.npy"))
    y_train = np.load(os.path.join(in_dir,"y_train.npy"))
    y_val = np.load(os.path.join(in_dir,"y_val.npy"))
    y_test = np.load(os.path.join(in_dir,"y_test.npy"))
    
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    
    x_train /= 255
    x_val /= 255
    x_test /= 255
    
    # Convert the images to grayscale for the CIFAR10 database
    datasets = [x_train,x_val,x_test]
    
    if database == 'CIFAR10':
        for i in range(len(datasets)):
            x = datasets[i]
            datasets[i] = 0.2989*x[:,:,:,0] + 0.5870*x[:,:,:,1] + 0.1140*x[:,:,:,2] 
        
    # Generate inputs
    if domain == 'Fourier':
        # Convert images to Fourier domain and use real and imaginary parts separately to deal with complex data
        x_train = np.zeros((datasets[0].shape[0],datasets[0].shape[1],datasets[0].shape[2],2))
        x_val = np.zeros((datasets[1].shape[0],datasets[1].shape[1],datasets[1].shape[2],2))
        x_test = np.zeros((datasets[2].shape[0],datasets[2].shape[1],datasets[2].shape[2],2))
        
        x_train[:,:,:,0] = np.real(np.fft.fftshift(np.fft.fft2(datasets[0]),axes=(-2,-1)))
        x_train[:,:,:,1] = np.imag(np.fft.fftshift(np.fft.fft2(datasets[0]),axes=(-2,-1)))
        x_val[:,:,:,0] = np.real(np.fft.fftshift(np.fft.fft2(datasets[1]),axes=(-2,-1)))
        x_val[:,:,:,1] = np.imag(np.fft.fftshift(np.fft.fft2(datasets[1]),axes=(-2,-1)))
        x_test[:,:,:,0] = np.real(np.fft.fftshift(np.fft.fft2(datasets[2]),axes=(-2,-1)))
        x_test[:,:,:,1] = np.imag(np.fft.fftshift(np.fft.fft2(datasets[2]),axes=(-2,-1)))
    else: #Sampling in image domain
        x_train = np.expand_dims(datasets[0],-1)
        x_val = np.expand_dims(datasets[1],-1)
        x_test = np.expand_dims(datasets[2],-1)
        
    # Generate targets
    if reconVSclassif == 'recon':
        y_train = np.expand_dims(datasets[0],-1)
        y_val = np.expand_dims(datasets[1],-1)
        y_test = np.expand_dims(datasets[2],-1)
  	
    else:    #classification
        num_classes = y_train.shape[-1]
    
    return [x_train, y_train, x_val, y_val, x_test, y_test]