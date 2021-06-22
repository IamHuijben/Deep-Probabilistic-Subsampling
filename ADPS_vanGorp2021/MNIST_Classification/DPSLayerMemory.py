"""
===================================================================================
    Source Name   : DPS_Layer_memory.py
    Description   : Layer that specifies DPS with memory of previously selected
                    samples, to clear this memory invoke the:
                    'initialize_sample_memory' method
===================================================================================
"""

# %% import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %% Create DPS Layer
class DPS(nn.Module):
       def __init__(self,args):
              super(DPS,self).__init__()
              
              # %% initialization
              # running paramters
              self.mux_in = args.mux_in #mux_in means: multiplexer_in
              self.mux_out = args.mux_out #mux_out means: multiplexer_out. We chose those names to indicate the amount of samples coming in (mux_in) and the amount coming out after subsampling (mux_out)
              self.temperature = args.temperature
              self.device = args.device
              self.batch_size = args.batch_size
              
              # create gumble noise constructor
              self.gumbel = torch.distributions.gumbel.Gumbel(0,1)
              
              # create the memorized samples
              self.sample_memory = 0
              
       def forward(self,logits):
              # %% forward call              
              #create perturbed logits
              if self.training == True:
                  noise = self.gumbel.sample((self.batch_size,self.mux_in,1)).repeat(1,1,self.mux_out).to(self.device)
              else:
                  noise = 0
              
              perturbed_logits = logits.unsqueeze(2).repeat(1,1,self.mux_out) + noise
              
              #masking from memory
              masking_matrix_memory = self.sample_memory.unsqueeze(2).repeat(1,1,self.mux_out) * -1000
              perturbed_logits += masking_matrix_memory
              
              #find the locations of the topk samples
              _,topk_indices = torch.topk(perturbed_logits[:,:,0],self.mux_out)
              
              # %% counting vectors (helpers)
              batch_indices = torch.tensor([range(self.batch_size)]).reshape(self.batch_size,1).repeat(1,self.mux_out)
              k_counter = torch.tensor([range(self.mux_out)])

              # %% create  hard sample mask
              hard_sample_mask = torch.zeros(self.batch_size,self.mux_in).to(self.device)
              hard_sample_mask[batch_indices,topk_indices] = 1
              
              # add the previous samples from memory
              hard_sample_mask += self.sample_memory
              
              # memorize this new sample matrix
              self.sample_memory = hard_sample_mask
              
              # %% create the soft sample matrix
              masking_matrix = torch.zeros(self.batch_size,self.mux_in,self.mux_out).to(self.device)
              

              # %% masking operaion:
              masking_matrix[batch_indices,topk_indices,k_counter] = -np.inf
              masking_matrix = torch.cumsum(masking_matrix, dim = 2)
              masking_matrix[batch_indices,topk_indices,k_counter] = 0 # mimic the exclusive=True parameter from tf.cumsum

              masked_logits = perturbed_logits + masking_matrix
              
              # softmaxing
              soft_sample_matrix = F.softmax(masked_logits/self.temperature,dim = 1)
              
              # collapse
              soft_sample_mask = torch.sum(soft_sample_matrix,dim=2)
              
              # %% Make sure that in the forward pass the hard sample matrix will be passed, whereas in the backwards pass, only the softsample matrix will be seen for the gradient. 
			  # The hard_sample_matrix already has no defined gradient, so doesn't need to be detached explicitly.
              sample_mask = hard_sample_mask - soft_sample_mask.detach() + soft_sample_mask 
              
              return sample_mask
       
       def initialize_sample_memory(self):
              self.sample_memory = torch.zeros(self.batch_size,self.mux_in).to(self.device)