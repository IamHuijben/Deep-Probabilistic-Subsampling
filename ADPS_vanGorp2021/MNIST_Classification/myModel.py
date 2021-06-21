"""
===================================================================================
    Source Name   : MyModel.py
    Description   : Specification of the models used for MNIST
===================================================================================
"""

# %% import dependancies
import torch
import torch.nn as nn

import DPSLayerMemory

# %% network
class Network(nn.Module):
    def __init__(self,args):
           # %% initialization
           super(Network,self).__init__()
           self.device = args.device
                                          
           # make distinction between DPS and A-DPS
           self.sampling = args.sampling
           
           if self.sampling == 'DPS':
                  self.no_iter = 1
                  self.logit_parameter = nn.Parameter(torch.randn(784)/4.)
                  args.mux_out = args.no_pixels
           elif self.sampling == 'ADPS':
                  self.no_iter = args.no_pixels
                  args.mux_out = 1
           else:
                  raise Exception('invalid sampling strategy selected, choose DPS or ADPS')
                  
           # create the layers as a module list
           self.f1 = nn.ModuleList([
                  nn.Sequential(
                  nn.Linear(784,784),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(p=0.3),
                  
                  nn.Linear(784,256),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(p=0.3),
               
                  nn.Linear(256,128),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(p=0.3),
               
                  nn.Linear(128,128),
                  nn.LeakyReLU(0.2),
                  ) for i in range(self.no_iter)])
           
           self.f2 = nn.ModuleList([
                  nn.Sequential(
                  nn.Linear(128,10),
                  ) for i in range(self.no_iter)])
           
           if self.sampling == 'ADPS':
                  self.g = nn.ModuleList([
                         nn.Sequential(
                         nn.Linear(128,256),
                         nn.LeakyReLU(0.2),
                         nn.Dropout(p=0.3),
                         nn.Linear(256,784),
                         ) for i in range(self.no_iter)])
                  
                  self.lstm = nn.LSTM(128,128,1)
        
           args.mux_in = 28**2
           self.DPS = DPSLayerMemory.DPS(args)
        
    # %% forward call    
    def forward(self, x):
           # get the batch size
           batch_size = x.size(0)
           
           # convert the input image into a vector
           x_vector = x.reshape(batch_size,784)
           
           #initialize the DPS memory
           self.DPS.initialize_sample_memory()
           
           # create the initial hidden states for the lstm
           h_var = torch.zeros(1,batch_size,128).to(self.device)
           c_var = torch.zeros(1,batch_size,128).to(self.device)
           lstm_in = torch.zeros(1,batch_size,128).to(self.device)
           
           # initalize all outputs for s_hat
           s_hat_all = torch.zeros(batch_size,10,self.no_iter).to(self.device)
           
           # iterate over the lstm
           for i in range(self.no_iter):
                  # generate logits from the hidden state
                  if self.sampling == 'DPS':
                         #DPS
                         logits = self.logit_parameter.unsqueeze(0).repeat(batch_size,1)
                  else:
                         #ADPS
                         _,(h_var,c_var) = self.lstm(lstm_in,(h_var,c_var))
                         logits = self.g[i](h_var.squeeze())
                     
                  # sampling
                  mask = self.DPS(logits)
                  y = mask*x_vector
                  
                  # encode
                  linear1_out = self.f1[i](y)
                  
                  # get the inputs ready for the lstm
                  lstm_in = linear1_out.unsqueeze(0)
                  
                  #last dense layer
                  s_hat = self.f2[i](linear1_out.squeeze())
                  
                  #save this result
                  s_hat_all[:,:,i] = s_hat
           
           return s_hat_all