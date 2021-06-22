"""
===================================================================================
    Source Name   : myModel.py
    Description   : Specification of the models used
===================================================================================
"""

# %% import dependencies
import torch
import torch.nn as nn

import fixedMasks
import fourierOperations
import DPSLayerMemory
import LOUPE

# %% Unfolded Proximal Gradient Network
class Proximal(nn.Module):
       def __init__(self):
              super(Proximal,self).__init__()
              # learning rates
              self.alpha0 = nn.Conv2d(1,1,3,padding = 1)
              self.alpha1 = nn.Conv2d(1,1,3,padding = 1)
              self.alpha2 = nn.Conv2d(1,1,3,padding = 1)
              
              # Proximals unfoldings
              self.prox0 = nn.Sequential(
                      nn.Conv2d(1,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,1,3,padding = 1),
                      )
              self.prox1 = nn.Sequential(
                      nn.Conv2d(1,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,1,3,padding = 1),
                      )
              
              self.prox2 = nn.Sequential(
                      nn.Conv2d(1,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,16,3,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(16,1,3,padding = 1),
                      )
              
       # %% forward call
       def forward(self,input_image,mask):
           # Creat state and image 0
           state_0 = self.alpha0(input_image)
           image_0 = self.prox0(state_0)
           
           # Creat state and image 1
           image_0_sampled = fourierOperations.Full_Map(image_0,mask)    
           state_1 = image_0 - self.alpha1(image_0_sampled) + state_0
           image_1 = self.prox1(state_1)
           
           # Creat state and image 2
           image_1_sampled = fourierOperations.Full_Map(image_1,mask)    
           state_2 = image_1 - self.alpha2(image_1_sampled) + state_0
           image_2 = self.prox2(state_2)
          
           # output the image
           return image_2
       
# %% Fixed Mask Network
class fixedMask(nn.Module):
       def __init__(self,args):
           super(fixedMask,self).__init__()
           self.device = args.device
           self.proximal = Proximal()
           self.mask = fixedMasks.createMask(args)
              
       def forward(self,input_image):
           #subsample
           subsampled_input = fourierOperations.Full_Map(input_image,self.mask)
           
           #reconstruct
           output_images = self.proximal(subsampled_input,self.mask)
           
           return output_images

# %% LOUPE Network
class LOUPEMask(nn.Module):
       def __init__(self,args):
           super(LOUPEMask,self).__init__()
           self.device = args.device
           self.proximal = Proximal()
           self.Pineda = args.Pineda
           self.batch_size = args.batch_size

           if self.Pineda == True:
               self.width = 368
               self.height = 640
           else:
               self.width = 208
               self.height = 208
               
           args.mux_out = args.no_lines 
           args.mux_in = self.width
           
           self.loupe = LOUPE.loupe(args)
           self.logit_parameter = nn.Parameter(torch.rand(args.mux_in)/5.)
              
       def forward(self,input_image):
           # get the mask from LOUPE
           mask = self.loupe(self.logit_parameter)
           mask = self.expandSampleMatrix(mask)
           
           #subsample
           subsampled_input = fourierOperations.Full_Map(input_image,mask)
           
           #reconstruct
           output_images = self.proximal(subsampled_input,mask)
           
           return output_images
       
       def expandSampleMatrix(self,mask):
            mask = mask.reshape(self.batch_size,1,self.width,1)
            mask = mask.repeat(1,1,1,self.height)
            return mask
        
# %% DPS Network
class DPSMask(nn.Module):
       def __init__(self,args):
           super(DPSMask,self).__init__()
           self.device = args.device
           self.proximal = Proximal()
           self.Pineda = args.Pineda
           self.batch_size = args.batch_size

           if self.Pineda == True:
               self.width = 368
               self.height = 640
           else:
               self.width = 208
               self.height = 208
               
           args.mux_out = args.no_lines 
           args.mux_in = self.width
           
           self.DPS = DPSLayerMemory.DPS(args)
           self.logit_parameter = nn.Parameter(torch.randn(args.mux_in)/4.)
              
       def forward(self,input_image):
           # clear sampling memory
           self.DPS.initialize_sample_memory()
           
           #check if the 30 DC lines are sampled
           if self.Pineda == True:
               lines = torch.arange(30)-15
               self.DPS.sample_memory[:,lines] = 1
           
           # unsqueeze the logits along the batch dimension
           logits = self.logit_parameter.unsqueeze(0).repeat(self.batch_size,1)
           
           # use the logits to create a sampling mask
           mask = self.DPS(logits)
           mask = self.expandSampleMatrix(mask)
           
           #subsample
           subsampled_input = fourierOperations.Full_Map(input_image,mask)
           
           #reconstruct
           output_images = self.proximal(subsampled_input,mask)
           
           return output_images
       
       def expandSampleMatrix(self,mask):
            mask = mask.reshape(self.batch_size,1,self.width,1)
            mask = mask.repeat(1,1,1,self.height)
            return mask
        
# %% ADPS Network
class ADPSMask(nn.Module):
       def __init__(self,args):
           super(ADPSMask,self).__init__()
           self.device = args.device
           self.Pineda = args.Pineda
           self.batch_size = args.batch_size

           if self.Pineda == True:
               self.width = 368
               self.height = 640
               self.no_iter = args.no_lines +1
           else:
               self.width = 208
               self.height = 208
               self.no_iter = args.no_lines 
               
           args.mux_out = 1
           args.mux_in = self.width
           
           self.no_iter = args.no_lines 
           
           self.DPS = DPSLayerMemory.DPS(args)
           
           # networks
           self.proximal = nn.ModuleList([
                   Proximal()
                   for i in range(self.no_iter)])
           
           self.lstm_size = 64
           
           self.SampleNet = nn.ModuleList([nn.Sequential(
                      nn.Conv2d(1,16,3),
                      nn.ReLU(),
                      nn.Conv2d(16,32,3),
                      nn.ReLU(),
                      nn.Conv2d(32,self.lstm_size,3),
                      nn.ReLU(),
                      nn.AdaptiveAvgPool2d(1),
                      nn.Flatten(),
                      )for i in range(self.no_iter)])
           self.lstm = nn.LSTM(self.lstm_size+self.width,self.lstm_size,num_layers=1)

           self.final_fc = nn.ModuleList([
                   nn.Linear(self.lstm_size,self.width)
                   for i in range(self.no_iter)])
                  
              
       def forward(self,input_image):
           # clear sampling memory
           self.DPS.initialize_sample_memory()
               
           # create an initial hidden state and context
           h_var = torch.zeros(1,self.batch_size,self.lstm_size).to(self.device)
           c_var = torch.zeros(1,self.batch_size,self.lstm_size).to(self.device)
           
           # initialize output images
           output_images = torch.zeros(self.batch_size,self.no_iter,self.width,self.height).to(self.device)
           current_output_image = 0
           
           # loop over all the iterations
           for i in range(self.no_iter):
               # creat the sampling mask from the logits
               logits = self.final_fc[i](h_var.reshape(self.batch_size,self.lstm_size))
               mask = self.DPS(logits)
            
               mask = self.expandSampleMatrix(mask)
               
               # begin by subsampling the input image,
               subsampled_input = fourierOperations.Full_Map(input_image,mask)
               
               # proximal mapping
               current_output_image = self.proximal[i](subsampled_input,mask)
                            
               #save the current output image
               output_images[:,i,:,:] = current_output_image[:,0,:,:]
               
               # create the next hidden state
               lines_chosen = mask[:,:,:,0].transpose(0,1)
               output_conv = self.SampleNet[i](current_output_image).unsqueeze(0)
               input_lstm = torch.cat((output_conv,lines_chosen),dim=2)
               _,(h_var,c_var) = self.lstm(input_lstm,(h_var,c_var))
          
           # output the images
           return output_images
       
       def expandSampleMatrix(self,mask):
            mask = mask.reshape(self.batch_size,1,self.width,1)
            mask = mask.repeat(1,1,1,self.height)
            return mask
        
# %% ADPS Network
class ADPSMask_legacy(nn.Module):
       def __init__(self,args):
           super(ADPSMask_legacy,self).__init__()
           self.device = args.device
           self.Pineda = args.Pineda
           self.batch_size = args.batch_size

           if self.Pineda == True:
               self.width = 368
               self.height = 640
               self.no_iter = args.no_lines +1
           else:
               self.width = 208
               self.height = 208
               self.no_iter = args.no_lines 
               
           args.mux_out = 1
           args.mux_in = self.width
           
           self.no_iter = args.no_lines 
           
           self.DPS = DPSLayerMemory.DPS(args)
           
           # networks
           self.proximal = nn.ModuleList([
                   Proximal()
                   for i in range(self.no_iter)])
           
           self.lstm_size = 64
           
           # slight error was made in intermediate size which we now hardcode to work with existing checkpoints
           self.SampleNet = nn.ModuleList([nn.Sequential(
                          nn.Conv2d(1,16,3),
                          nn.ReLU(),
                          nn.Conv2d(16,30,3),
                          nn.ReLU(),
                          nn.Conv2d(30,self.lstm_size,3),
                          nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(),
                          )for i in range(self.no_iter)])
           self.lstm = nn.ModuleList([nn.LSTM(self.lstm_size,self.lstm_size,num_layers=1)for i in range(self.no_iter)])
               
           self.final_fc = nn.ModuleList([
                   nn.Linear(self.lstm_size,self.width)
                   for i in range(self.no_iter)])
                  
              
       def forward(self,input_image):
           # clear sampling memory
           self.DPS.initialize_sample_memory()
           
           #check if the 30 DC lines are sampled
           if self.Pineda == True:
               lines = torch.arange(30)-15
               self.DPS.sample_memory[:,lines] = 1
               
           # create an initial hidden state and context
           h_var = torch.zeros(1,self.batch_size,self.lstm_size).to(self.device)
           c_var = torch.zeros(1,self.batch_size,self.lstm_size).to(self.device)
           
           # initialize output images
           output_images = torch.zeros(self.batch_size,self.no_iter,self.width,self.height).to(self.device)
           current_output_image = 0
           
           # loop over all the iterations
           for i in range(self.no_iter):
               # creat the sampling mask from the logits
               logits = self.final_fc[i](h_var.reshape(self.batch_size,self.lstm_size))
               mask = self.DPS(logits)
            
               mask = self.expandSampleMatrix(mask)
               
               # begin by subsampling the input image,
               subsampled_input = fourierOperations.Full_Map(input_image,mask)
               
               # proximal mapping
               current_output_image = self.proximal[i](subsampled_input,mask)
                            
               #save the current output image
               output_images[:,i,:,:] = current_output_image[:,0,:,:]
               
               # create the next hidden state
               input_lstm = self.SampleNet[i](current_output_image).unsqueeze(0)
               _,(h_var,c_var) = self.lstm[i](input_lstm,(h_var,c_var))
          
           # output the images
           return output_images
       
       def expandSampleMatrix(self,mask):
            mask = mask.reshape(self.batch_size,1,self.width,1)
            mask = mask.repeat(1,1,1,self.height)
            return mask
                  