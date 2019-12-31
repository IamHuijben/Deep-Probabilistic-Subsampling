"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : inference.py
                    Script to run inference on a trained model
                    
    Author        : Iris Huijben
    Date          : 02/08/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019
==============================================================================
"""

import os
import numpy as np
import tensorflow as tf
from keras.models import Model
import myModels
from matplotlib import pyplot as plt
from pathsetupCIFAR10 import in_dir
import dataLoader
#=============================================================================

#Indicate the name of the model, and the name of the corresponding weight file 
versionName ="5_MNIST_32_loupe_classify_image_fact8_l2pen0_lr0.0002_EM_0-0.0005-0-50_patience10_cooldown20"
weightFile = "weights-36-0.06"

savedir = os.path.join(os.path.dirname(__file__),versionName,weightFile+".hdf5")

#########################################################################
# Deduce parameters from versionName
#########################################################################

indComp = versionName.find('fact')
try:
    comp = int(versionName[indComp+4:indComp+6])
except:
    comp = int(versionName[indComp+4:indComp+5])

if versionName.find("classif") > -1:
    reconVSclassif = 'classif'
else:
    reconVSclassif = 'recon'

if versionName.find("CIFAR10") > -1:
    database = "CIFAR10"
else:
    database = "MNIST"
    
if versionName.find("image") > -1:
    imageDomainSampling = True
    domain = 'image'
elif versionName.find("Fourier") > -1:
    imageDomainSampling = False
    domain = 'Fourier'
else: #Use Fourier as default domain if not indicated in the versionname
    imageDomainSampling = False;
    domain = 'Fourier' 
    print('Default sampling domain Fourier selected')
 
    
if versionName.find("topK") > -1:
    gumbelTopK = True
    DPSsamp = True
else:
    gumbelTopK = False
    
circle = False              # In case DPSsamp is False, we use a non-trainable (fixed) sampling pattern which is either uniform, circular or random. Only implemented for comp=4 or comp=8
DPSsamp = False            # If true, sub-sampling is learned by DPS. If false, we use a fixed sub-sampling pattern (uniform or random)
Bahadir = False
uniform = False

if versionName.find("DPS") > -1:
    DPSsamp = True            
elif versionName.find("loupe") > -1:
    Bahadir = True
elif versionName.find("uniform") > -1:
    uniform = True             
elif versionName.find("lpf") > -1:
    circle = True            
else:
    random = True
    
print('Detected task and database: ', reconVSclassif, database)
print('Sampledomain: ', domain, ',with comp = ', comp)


#=============================================================================
#%%
"""
=============================================================================
    Load and prepare the datasets
=============================================================================
"""
[_, _, _, _, x_test, y_test] = dataLoader.LoadData(database, domain, reconVSclassif)
input_dim = x_test.shape[1:]
target_dim = y_test.shape[1:]
num_classes = target_dim[-1]

#%%
"""
=============================================================================
    Parameter definitions
=============================================================================
"""

# Model params
mux_out = int(np.ceil(((input_dim[0])*(input_dim[1]))//comp))     # Multiplexer output dims: the amount of samples to be sampled from the input   
folds = 5                   # Amount of unfoldings for the reconstruction part
n_convs = 6                 # number of convolutional layers in the prox
OneOverLmult = 0            # The stepsize in the unfolded proximal gradient scheme. If 0; use a convolutional kernel instead.
share_prox_weights = False  # Boolean whether weights are shared over the unfoldings in the proximal mapping
batch_size = 32

# Some params that do not matter for inference
n_epochs = 0                                                        
batchPerEpoch = np.int32(np.ceil(x_test.shape[0]/batch_size))          
learningrate = 0
subSampLrMult = 0    
tempIncr = 1         


"""
=============================================================================
    Model definition
=============================================================================
"""
if database == 'CIFAR10' and reconVSclassif == 'recon':
    D_weight= 0.004 
    loss_weights = [1,D_weight]

if not Bahadir:
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
                                [],
                                subSampLrMult,
                                DPSsamp,
                                Bahadir,
                                uniform,
                                circle,
                                [],
                                folds,
                                reconVSclassif,
                                0,
                                batchPerEpoch,
                                share_prox_weights = share_prox_weights,
                                OneOverLmult = OneOverLmult, # 0 means train a convolutional kernel instead
                                n_convs = n_convs,
                                learningrate=0.0002,
                                type_recon=type_recon,
                                gumbelTopK=gumbelTopK)


    preloaded_layers = model.layers.copy()
    preloaded_weights = []
    for pre in preloaded_layers:
        preloaded_weights.append(pre.get_weights())
    
    model.load_weights(os.path.join(savedir,".hdf5"),by_name=True)
    
    # compare previews weights vs loaded weights to check that loading went correctly
    for layer, pre in zip(model.layers, preloaded_weights):
        weights = layer.get_weights()
    
        if weights:
            if np.array_equal(weights, pre):
                print('not loaded', layer.name)
            else:
                print('loaded', layer.name)
                
else: #In case of LOUPE (Bahadir, 2019), load the model with a thresholded sample mask
    modelParams = [reconVSclassif, share_prox_weights, n_convs, OneOverLmult,folds]
    import ThresholdingLOUPE
    model = ThresholdingLOUPE.InferenceModelLOUPE(comp,input_dim,database, domain,modelParams, num_classes, savedir,weightFile)
    
 
def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1)

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)

if reconVSclassif == 'recon':
    loss = 'mean_squared_error' 
    metrics = [SSIM, PSNR,'mean_squared_error']
else:
    loss = 'categorical_crossentropy' 
    metrics = ['accuracy']

model.compile('adam',loss,metrics)
model.summary()

#%%
"""
=============================================================================
    Inference
=============================================================================
"""
pred = model.predict(x_test)

"""
=============================================================================
    Evaluate & Display
=============================================================================
"""
savefigs = False

if reconVSclassif == 'recon':
    
    # Evaluate the metrics
    loss,SSIM,PSNR,MSE = model.evaluate(x_test,y_test)

    nr_examples = len(x_test)
    print("MSE across {} examples: {}".format(nr_examples,MSE))
    print("PSNR across {} examples: {}".format(nr_examples,PSNR))
    print("SSIM across {} examples: {}".format(nr_examples,SSIM))

    with open(savedir+"\\results.txt", "w") as text_file:
        print("MSE across {} examples: {} \n".format(nr_examples,MSE),file=text_file)
        print("PSNR across {} examples: {} \n".format(nr_examples,PSNR),file=text_file)
        print("SSIM across {} examples: {}".format(nr_examples,SSIM),file=text_file)


    # Display predictions
    disp_examples = [5,59,16]
    for disp_example in disp_examples:
        plt.figure()
        spect =np.sqrt(x_test[disp_example,:,:,0]**2+x_test[disp_example,:,:,1]**2)
        plt.imshow(20*np.log10(0.001+spect/np.max(spect)), cmap='jet',vmin=-40,vmax=0)
        plt.axis('off')
        
        if savefigs:
            plt.savefig(savedir+'\\example_{}_input_40dB.svg'.format(disp_example),bbox_inches='tight')
        plt.pause(.1)
               
        plt.figure()
        plt.imshow(y_test[disp_example,:,:,0], cmap='gray',vmin=0,vmax=np.max(y_test[disp_example]))
        plt.axis('off')
        
        if savefigs:
            plt.savefig(savedir+'\\example_{}_target.svg'.format(disp_example),bbox_inches='tight')
        plt.pause(.1)
        
        plt.figure()
        plt.imshow(pred[disp_example,:,:,0], cmap='gray',vmin=0,vmax=np.max(y_test[disp_example]))
        plt.axis('off')
        
        if savefigs:
            plt.savefig(savedir+'\\example_{}_prediction.svg'.format(disp_example),bbox_inches='tight')
        plt.pause(.1)

else: #Classification
    if DPSsamp:
        accuracies = []
        for i in range(0,100):
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            accuracies.append(accuracy)
        accuracy = np.mean(accuracies)
        std = np.std(accuracies)
    else:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        

    print('Model: ', versionName)
    print('Test accuracy: ', accuracy)
    if DPSsamp:
        print('Test error: {}+-{}'.format((1-accuracy)*100,np.std((1-np.array(accuracies))*100)))
            
        with open(savedir+"\\results.txt", "w") as text_file:
            print('Model: ', versionName,file=text_file)
            print('Weights: ', weightFile, file=text_file)
            print('Test accuracy: ', accuracy,file=text_file)
            print('Test error: {}+-{}'.format((1-accuracy)*100,np.std((1-np.array(accuracies))*100)),file=text_file)
    else:
        print('Test error: {}'.format((1-accuracy)*100))
        
        with open(savedir+"\\results.txt", "w") as text_file:
            print('Model: ', versionName,file=text_file)
            print('Weights: ', weightFile, file=text_file)
            print('Test accuracy: ', accuracy,file=text_file)
            print('Test error: {}'.format((1-accuracy)*100), file=text_file)


#% Display realizations of sampling schemes
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
     
     print('thresholded mask used during training')
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

   
# Plot one realization
print('One realization')
plt.figure()
plt.imshow(pattern,cmap='gray_r', vmin=0, vmax=1)
plt.axis('off')
if savefigs:
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
if DPSsamp:
    print('Trained distribution')
    plt.figure()
    plt.imshow(distribution,cmap='gray_r')
    plt.axis('off')
    if savefigs:
        plt.savefig(savedir+'\hardSamples_dist.svg',bbox_inches='tight')
    plt.pause(.1)
