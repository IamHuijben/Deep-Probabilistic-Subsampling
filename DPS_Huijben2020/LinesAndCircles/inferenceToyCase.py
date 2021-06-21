"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : inferenceToyCase.py
                    This file load weights of a pretrained model and runs inference
                    
    Author        : Iris Huijben
    Date          : 09/08/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019

==============================================================================
"""

import sys, os.path as path, os

import numpy as np
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Model

from sklearn.model_selection import train_test_split
import myModel

#=============================================================================
versionName = "Toy_loupe_recon_fact32_lr0.0001_lrMult10_EM_0.0001-0.0005-0-50"
weightFile = "weights-833-0.03"
savedir = os.path.join(os.path.dirname(__file__),versionName)

indComp = versionName.find('fact')
try:
    comp = int(versionName[indComp+4:indComp+6])
except:
    comp = int(versionName[indComp+4:indComp+5])

    
circle = False
DPSsamp = False
Bahadir = False
uniform = False

if versionName.find("GumbelTopK") > -1:
    gumbelTopK = True
    DPSsamp = True
else:
    gumbelTopK = False


if versionName.find("DPS") > -1:
    DPSsamp = True            # If true, sub-sampling is learned by LASSY. If false, we use a fixed sub-sampling pattern (uniform or random)
elif versionName.find("loupe") > -1:
    Bahadir = True
elif versionName.find("uniform") > -1:
    uniform = True             # In case DPSsamp is False, we use a non-trainable (fixed) sampling pattern which is either uniform, circular or random  
elif versionName.find("lpf") > -1:
    circle = True             # In case DPSsamp is False, we use a non-trainable (fixed) sampling pattern which is either uniform, circular or random
 

#=============================================================================

#%%
"""
=============================================================================
    Load testset
=============================================================================
"""
input_size = (32,32)
nr_examples = 1000

test_x = np.load('testSet.npy')
test_y = np.load('testSetY.npy')
#disp_example = 11
#plt.imshow(test_y[disp_example,:,:,0])

#
    #%%
"""
=============================================================================
    Parameter definitions
=============================================================================
"""

input_dim = [input_size[0],input_size[1],2]           # Dimensions of the inputs to the network: [fourier bins, IQ components]
target_dim = [input_size[0],input_size[1],2]          # Dimension of the targets for the network: [time steps]
mux_out = np.prod(input_dim[0:2])//comp      # Multiplexer output dims: the amount of samples to be sampled from the input


"""
=============================================================================
    Model definition
=============================================================================
"""
def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1)

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)

loss = 'mean_squared_error' 
metrics = [SSIM, PSNR,'mean_squared_error']

if not Bahadir:
    model = myModel.full_model(
                        input_dim, 
                        target_dim, 
                        comp,
                        mux_out, 
                        2, 
                        [], 
                        DPSsamp, 
                        Bahadir,
                        uniform, 
                        circle, 
                        1000, 
                        32, 
                        gumbelTopK)
    
    ## Print model summary:
    model.load_weights(os.path.join(savedir,weightFile+".h5"))
    model.compile(optimizer='adam',loss=loss, metrics=metrics)


else:
    import ThresholdingLOUPE
    model = ThresholdingLOUPE.LoadModelLOUPE(comp,input_dim,savedir,weightFile)
    
    sgd = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss=loss, metrics=metrics)

model.summary()


#%%
"""
=============================================================================
    Inference
=============================================================================
"""
pred = model.predict(test_x)



#%%
"""
=============================================================================
    Evaluate
=============================================================================
"""
def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=10)

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=10)

loss = 'mean_squared_error' 
metrics = [SSIM, PSNR,'mean_squared_error']
model.compile('adam',loss,metrics)


loss,SSIM,PSNR,MSE = model.evaluate(test_x,test_y)

print("MSE across {} examples: {}".format(nr_examples,MSE))
print("PSNR across {} examples: {}".format(nr_examples,PSNR))
print("SSIM across {} examples: {}".format(nr_examples,SSIM))

with open(savedir+"\\results.txt", "w") as text_file:
    print("MSE across {} examples: {} \n".format(nr_examples,MSE),file=text_file)
    print("PSNR across {} examples: {} \n".format(nr_examples,PSNR),file=text_file)
    print("SSIM across {} examples: {}".format(nr_examples,SSIM),file=text_file)

#%%
"""
=============================================================================
    Display
=============================================================================
"""
savefigs = True

#%%
"""
=============================================================================
    Display
=============================================================================
"""
savefigs = True

disp_examples = [18, 60]

for i in range(len(disp_examples)):
    disp_example = disp_examples[i]
    plt.figure()
    spect =np.sqrt(test_x[disp_example,:,:,0]**2+test_x[disp_example,:,:,1]**2)
    plt.imshow(20*np.log10(0.001+spect/np.max(spect)), cmap='jet',vmin=-40,vmax=0)
    plt.axis('off')
    
    if savefigs:
        plt.savefig(savedir+'\\example_{}_input_40dB.png'.format(disp_example),bbox_inches='tight')
        plt.savefig(savedir+'\\example_{}_input_40dB.svg'.format(disp_example),bbox_inches='tight')
    plt.pause(.1)
    
    
    plt.figure()
    plt.imshow(test_y[disp_example,:,:,0], cmap='gray',vmin=0,vmax=np.max(test_y[disp_example]))
    plt.axis('off')
    plt.pause(.5)
    
    if savefigs:
        plt.savefig(savedir+'\\example_{}_target.png'.format(disp_example),bbox_inches='tight')
        plt.savefig(savedir+'\\example_{}_target.svg'.format(disp_example),bbox_inches='tight')
    plt.pause(.1)
    
    plt.figure()
    plt.imshow(pred[disp_example,:,:,0], cmap='gray',vmin=0,vmax=np.max(test_y[disp_example]))
    plt.axis('off')
    
    if savefigs:
        plt.savefig(savedir+'\\example_{}_prediction.png'.format(disp_example),bbox_inches='tight')
        plt.savefig(savedir+'\\example_{}_prediction.svg'.format(disp_example),bbox_inches='tight')
    plt.pause(.1)

#%%
#% Display samples
n_MC = 1000
if DPSsamp:
    model_sampling =  Model(inputs = model.input, outputs = model.get_layer("AtranA_0").output)    
    patterns = model_sampling.predict_on_batch(tf.zeros((n_MC,32,32,2)))[:,:,:,0]
    pattern = patterns[0]

    model_distribution  =  Model(inputs = model.input, outputs = model.get_layer("CreateSampleMatrix").output)
    logits = model_distribution.predict_on_batch(tf.zeros((1,32,32,2)))
    
    unnormDist = np.exp(logits)
    distribution = np.transpose(np.transpose(unnormDist) / np.sum(unnormDist,1))
    distribution = np.reshape(np.sum(distribution,axis=0),(input_dim[0],input_dim[1]))
    
   
elif Bahadir:
     model_sampling =  Model(inputs = model.input, outputs = model.get_layer("HardSampleMask").output)     
     patterns = []
 
     Mask = model_sampling.predict_on_batch(tf.zeros((1,input_dim[0],input_dim[1],input_dim[2])))
     #Note order of distribution and pattern in Mask variable is switches due to the fftshift function
     pattern = Mask[0]
     renormMask = Mask[1]
     
     thresh = np.random.uniform(0.0,1.0,(input_dim[0],input_dim[1]))
     sampleMask = ThresholdingLOUPE.sigmoid(12 * (renormMask-thresh))
     
     plt.figure()
     plt.imshow(sampleMask, cmap="hot_r")
     plt.xticks([])
     plt.yticks([])
     plt.savefig(os.path.join(savedir,'NonHardSamplesTraining.svg'), bbox_inches="tight")
     #plt.colorbar(shrink=0.5)
     plt.pause(.1)
     
     def MCsamplingLOUPE(renormMask):
         MCsamples = []
         
         #We explicitly have to run the thresholding code here again, as otherwise we do not use different uniform noise every MC sampling
         for i in range(n_MC):
             
            thresh = np.random.uniform(0.0,1.0,(input_dim[0],input_dim[1]))
            sampleMask = ThresholdingLOUPE.sigmoid(12 * (renormMask-thresh))
            
            # Make sure to only select M hard samples
            sampleCoord = ThresholdingLOUPE.largest_indices(sampleMask,mux_out)
            hardSamples = np.zeros_like(sampleMask)
            hardSamples[sampleCoord[0],sampleCoord[1]] = 1
             
            MCsamples.append(hardSamples)
         return np.stack(MCsamples,axis=0)
     
     patterns = MCsamplingLOUPE(renormMask)
             
        
else:
     model_sampling =  Model(inputs = model.input, outputs = model.get_layer("CreateSampleMatrix").output)
     pattern = np.expand_dims(model_sampling.predict_on_batch(tf.zeros((n_MC,32,32,2))),0)[0,:,:,0]
#%%    

# Plot one realization
print('One realization')
plt.imshow(pattern,cmap='gray_r', vmin=0, vmax=1)
plt.axis('off')
if savefigs:
    plt.savefig(savedir+'\hardSamples.png',bbox_inches='tight')
    plt.savefig(savedir+'\hardSamples.svg',bbox_inches='tight')
    plt.pause(.1)

# Plot MC plots
if DPSsamp or Bahadir:   
    print(str(n_MC)+'times MC sampling')
    plt.figure()
    plt.imshow(-np.mean(patterns,0),cmap='gray')
    plt.axis('off')
    
    if savefigs:
        plt.savefig(savedir+'\hardSamples_MCdist.svg',bbox_inches='tight')
        plt.pause(.1)

# Plot trained distribution
if DPSsamp or Bahadir:
    print('Trained distribution')
    plt.figure()
    plt.imshow(distribution,cmap='gray_r')
    plt.axis('off')
    if savefigs:
        plt.savefig(savedir+'\hardSamples_dist.svg',bbox_inches='tight')
        plt.pause(.1)

   
