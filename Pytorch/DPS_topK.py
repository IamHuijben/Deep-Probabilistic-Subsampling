"""
===================================================================================
    Eindhoven University of Technology
===================================================================================
    Source Name   : DPS_topK.py
                    Pytorch Implementation of the Top-K DPS strategy from:
                    Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019
    Author        : Hans van Gorp
    Date          : 27/04/2020
===================================================================================
To Initialize the DPS layer a couple of parameters are required:
       ---------------------------------------------------------------------------|
       | Name        | Description                                                |
       ---------------------------------------------------------------------------|
       | mux in      | Dimensionality of input vector                             |
       |             |                                                            |
       | mux out     | Number of samples to take                                  |
       |             |                                                            |
       | temperature | Temperature parameter of the gumbel-softmax trick.         |
       |             |   														  |
       |             |                                                            |
       | device      | device at which input and output tensors are located       |
       |             | e.g. "cpu" or "device(type='cuda', index=0)"               |
       ---------------------------------------------------------------------------|
===================================================================================
The input output relation:
       
Input:  tensor x of dimensionality [batch size, mux in] 
		This implementation deals with a batch of vectors x. In case x is higher dimensional, the code for the sampling mask should be extended with these higher dimensions. 
		However, typically you would still like to sample over one dimension at a time so the mask can best be copied for the extra dimensions. 
		In case you want to sample a 2D image space, the easiest way to use this function is to first vectorize the image.
Output: zero-padded subsampled tensor y of dimensionality [batch size, mux in], 
        with only mux out amount of non-zero elements
===================================================================================
"""

# %% import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %% Create DPS Layer
class DPS(nn.Module):
       def __init__(self,parameters):
              super(DPS,self).__init__()
              
              # %% initialization
              # running paramters
              self.mux_in = parameters['mux in'] #mux_in means: multiplexer_in
              self.mux_out = parameters['mux out'] #mux_out means: multiplexer_out. We chose those names to indicate the amount of samples coming in (mux_in) and the amount coming out after subsampling (mux_out)
              self.temperature = parameters['temperature']
              self.device = parameters['device']
              self.compression = parameters['compression'] # The factor with which the subsample the data
              
              # create gumble noise constructor
              self.gumbel = torch.distributions.gumbel.Gumbel(0,1)
              
              # create the logit parameter
              self.logits = nn.Parameter(torch.randn(self.mux_in,)/self.compression)
              
              
       def forward(self,x):
              # x is the input of the actual input of the entire model. It directly enters this sampling layer.
              # %% forward call
              #find the batch size
              batch_size = x.size(0)
              
              #create perturbed logits
              noise = self.gumbel.sample((batch_size,self.mux_in,1)).repeat(1,1,self.mux_out).to(self.device)
              perturbed_logits = torch.transpose(self.logits.repeat(batch_size,self.mux_out,1),1,2) + noise
              
              #find the locations of the topk samples
              _,topk_indices = torch.topk(perturbed_logits[:,:,0],self.mux_out)
              
              # %% counting vectors (helpers)
              batch_indices = torch.tensor([range(batch_size)]).reshape(batch_size,1).repeat(1,self.mux_out)
              k_counter = torch.tensor([range(self.mux_out)])

              # %% create  hard sample matrix
              hard_sample_matrix = torch.zeros(batch_size,self.mux_in).to(self.device)
              hard_sample_matrix[batch_indices,topk_indices] = 1
              
              # %% create the soft sample matrix
              masking_matrix = torch.zeros(batch_size,self.mux_in,self.mux_out).to(self.device)

              #cumulative masking:
			  # note that the masking in the pytorch implementation is done in the logits space. The tf implementation does this in the probability space.
			  # It should however not make a great difference, it is just a different implementation of the same masking behaviour.
              masking_matrix[batch_indices,topk_indices,k_counter] = -np.inf
              masking_matrix = torch.cumsum(masking_matrix, dim = 2)
              masking_matrix[batch_indices,topk_indices,k_counter] = 0 # mimic the exclusive=True parameter from tf.cumsum
              
              masked_logits = perturbed_logits + masking_matrix
              
              # softmaxing
              soft_sample_matrix = F.softmax(masked_logits/self.temperature,dim = 1)
              
              # collapse
              soft_sample_matrix_collapsed = torch.sum(soft_sample_matrix,dim=2)
              
              # %% Make sure that in the forward pass the hard sample matrix will be passed, whereas in the backwards pass, only the softsample matrix will be seen for the gradient. 
			  # The hard_sample_matrix already has no defined gradient, so doesn't need to be detached explicitly.
              sample_matrix = hard_sample_matrix - soft_sample_matrix_collapsed.detach() + soft_sample_matrix_collapsed 
              
              # %% element-wise multiplication of the input tensor with the binary mask. The non-sampled values are now set at zero.
              y = sample_matrix * x
			  
			  # Note that this layer outputs the already subsampled (and zero-filled) input signal. This is different than the output of the tf implemenation of the topKsampling layer.
			  # That layer outputs a hard (or for back prop a soft) sampling matrix of size MxN. Only after this layer a sampling mask is created from that and element-wise applied on the input data.

              return y