"""
===================================================================================
    Source Name   : LOUPE.py
    Description   : Pytorch Implementation of LOUPE from Bahadir et al. 2020
                    See https://github.com/cagladbahadir/LOUPE/ for their
                    original TensorFlow implementation 
===================================================================================
"""

# %% import dependencies
import torch
import torch.nn as nn

# %% Create loupe Layer
class loupe(nn.Module):
       def __init__(self,args):
              super(loupe,self).__init__()
              self.hard = False
              self.batch_size = args.batch_size
              
              
              self.device = args.device
              
       def forward(self,logits):
           # go from logits to probabilities
           prob_mask = torch.sigmoid(5* logits)
           
           #rescale
           xbar = torch.mean(prob_mask)
           r = 0.125/xbar
           beta = 0.875/(1-xbar)
           
           #rescaling if different depending on xbar>M/N or xbar<M/N
           if xbar>0.125:
               scaled_mask = prob_mask*r
           else:
               scaled_mask = 1 - (1-prob_mask)*beta
               
           #create random uniform thresholds as reparameterization trick
           thresholds = torch.rand(208).to(self.device)
           
           #if hard thresholding select the 26 biggest p
           if self.hard == True:
               sample_mask = torch.zeros(208).to(self.device)
               _,max_indices = torch.topk(scaled_mask-thresholds,26)
               sample_mask[max_indices] = 1
           else:
           #when softsampling use a sigmoid as relaxation
               sample_mask = torch.sigmoid(12*(scaled_mask -thresholds))
           
           
           return sample_mask.unsqueeze(0).repeat(self.batch_size,1)
       