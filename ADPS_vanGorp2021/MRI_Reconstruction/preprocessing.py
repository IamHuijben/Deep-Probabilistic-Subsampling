"""
===================================================================================
    Source Name   : preprocessing.py
    Description   : Script used to unpack and preprocess the raw MRI data
===================================================================================
"""
# %% import dependencies
import os
import h5py
from pathlib import Path
import numpy as np
import argparse
import torch

# %% my preprocessing functions
def readAndCropMRIimages(path, prevRes, newRes, innerVol):
    print("Read and crop images...")    
    xData = []
    i = 0
    #Load mri images (targets)
    for file in os.listdir(path):   
            
            print('Processing file: '+ str(i), end='\r')
            full_file = Path(path,file)
            data = h5py.File(full_file, 'r')
            
            esc_target = data[list(data.keys())[2]]
            # Only take the innerVol perc. of inner slices
            nrSlices = np.size(esc_target,0)
            innerSlices = int(innerVol*nrSlices)
            esc_targetInnerVol = esc_target[innerSlices//2:(nrSlices-(innerSlices//2)+1)]
            
            
            # find if something funny is going on with the resolution
            size1 = len(esc_targetInnerVol[0,])
            size2 = len(esc_targetInnerVol[0,0,])
            
            prevRes1 = size1
            prevRes2 = size2
            
            halfDiff1 = (prevRes1 - newRes)/2
            halfDiff2 = (prevRes2 - newRes)/2
            
            
            # Crop the MRI images to the central newRes^2 number of pixels
            esc_target_crop = esc_targetInnerVol[:,int(halfDiff1):int(prevRes1-halfDiff1),int(halfDiff2):int(prevRes2-halfDiff2)]
            
            # tranpose
            esc_target_transposed = np.transpose(esc_target_crop,(0,2,1))
            
            # append
            xData.append(esc_target_transposed)
            
            
            i = i+1
   
    xData = np.concatenate(xData,axis=0) #[samples,X,Y]
    print('')
    return xData

def normMRIim(xData):
    print("Normalizing images...")
    minVal = 0
    maxVal = 4e-4
    
    xDataNorm = (xData-minVal)/(maxVal-minVal)

    return xDataNorm

def saveSlices(savedir, datasetNameX, xData):
    print("Saving " + datasetNameX+"...")

    hf = h5py.File(os.path.join(savedir,datasetNameX+'.h5'), 'w')
    hf.create_dataset(datasetNameX, data=xData)
    hf.close()
        
    return

def TrainValSplit(xData,perm, totalNrSamples):
    print("Split train and validation set...")

    trainSlices = totalNrSamples[0]
    valSlices = totalNrSamples[1]
    
    xDataShuffled = xData[perm]

    x_train= xDataShuffled[:trainSlices]
    x_val = xDataShuffled[trainSlices:(trainSlices+valSlices)] 
    return x_train, x_val

# %% Pineda et al. reading functions
def check_if_string_in_file(file_name, string_to_search):
    """ Check if any line in the file contains given string """
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            if string_to_search in line:
                return True
    return False


def Inverse_Fourier(k_space):
    complex_pixels = torch.ifft(k_space,2)
    envelop_pixels = torch.sqrt(complex_pixels[:,:,:,0]**2 + complex_pixels[:,:,:,1]**2)
    return envelop_pixels

def Forward_Fourier(pixels):
    k_space = torch.rfft(pixels,2,onesided=False)
    return k_space


def readAndCropMRIimages_Pineda(path,split):
    print("Read and crop images following Pineda et al...")    
    xData = []
    i = 0
    #Load mri images (targets)
    for file in os.listdir(path):
            #check if we need to account for the datasplit
            if split == 'val':
                if check_if_string_in_file("data\\val_Pineda.txt",file) == False:
                    continue
            if split == 'test':
                if check_if_string_in_file("data\\test_Pineda.txt",file) == False:
                    continue

            
            
            print('Processing file: '+ str(i), end='\r')
            full_file = Path(path,file)
            data = h5py.File(full_file, 'r')
            
            kspace_data = data[list(data.keys())[1]][:]
            kspace_data = torch.from_numpy(np.stack([kspace_data.real, kspace_data.imag], axis=-1))
            
            # make sure that all kspaces are of the same size (slices,640,400)
            new_size = 368
            
            k_space = torch.zeros((kspace_data.size(0),640,new_size,2))
            
            old_size = kspace_data.size(2)
            
            difference = new_size - old_size
            
            if difference>=0:
                k_space[:,:,int(difference/2):old_size+int(difference/2),:] = kspace_data
            else:
                k_space = kspace_data[:,:,int(-difference/2):new_size+int(-difference/2),:]
            
            # create the target
            target = Inverse_Fourier(k_space)
            
            # shift it in place
            target = torch.roll(target,shifts = (320,200),dims = (1,2)).transpose(1,2)
            
            # normalize using mean value
            target /= 7.072103529760345e-07
            
            
            # append
            xData.append(target)
            
            i = i+1
   
    xData = np.concatenate(xData,axis=0) #[samples,X,Y]
    
    return xData

# %% main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path',type=str,help='specify the path to the location of the MRI knee data')
    parser.add_argument('-Pineda',type=bool,help='preprocess in the same manner as Pineda et al.',default=True)
    args = parser.parse_args()
    
    assert args.path != None
    
    # %% preprocessing in my way
    if args.Pineda == False:
        # Preprocess train and val set
        prevRes =  320  #original resolution of MRI images
        newRes = 208    #resolution after cropping to the central pixels
        innerVol = 0.5  #Percentage of inner slices that is selected from the 
        
        xData = readAndCropMRIimages(args.path+"\\knee_singlecoil_train", prevRes, newRes, innerVol)
        xDataNorm = normMRIim(xData)
        
        totalNrSamples = [8000, 2000]
        perm = np.load('data//PermTrainVal.npy')
        x_train, x_val = TrainValSplit(xDataNorm, perm, totalNrSamples)
        
        savedir = 'data//preprocessed'
        saveSlices(savedir, 'x_train', x_train)
        saveSlices(savedir, 'x_val', x_val)
        
        
        # Preprocess test set
        xData = readAndCropMRIimages(args.path+"\\knee_singlecoil_val", prevRes, newRes, innerVol)
        xDataNorm = normMRIim(xData)
        
        totalNrSamples = 3000
        perm = np.load('data//PermTest.npy')
        x_test = xDataNorm[perm[0:totalNrSamples]]
        
        savedir = 'data//preprocessed'  
        saveSlices(savedir, 'x_test', x_test)
        
    # %% preprocessing in the manner of Pineda et al.
    else:
        # Preprocess train set following Pineda et al.
        xData_train = readAndCropMRIimages_Pineda(args.path+"\\knee_singlecoil_train",'train')

        savedir = 'data\\preprocessed_Pineda'  
        saveSlices(savedir, 'x_train', xData_train)
        
        # Preprocess validation set following Pineda et al.
        xData_val = readAndCropMRIimages_Pineda(args.path+"\\knee_singlecoil_val",'val')
        
        savedir = 'data\\preprocessed_Pineda'  
        saveSlices(savedir, 'x_val', xData_val)
        # Preprocess test set following Pineda et al.
        xData_test = readAndCropMRIimages_Pineda(args.path+"\\knee_singlecoil_val",'test')
        
        savedir = 'data\\preprocessed_Pineda'  
        saveSlices(savedir, 'x_test', xData_test)
