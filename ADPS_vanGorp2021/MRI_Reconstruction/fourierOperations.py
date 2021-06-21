
"""
===================================================================================
    Source Name   : Fourier_Operations.py
    Description   : Some Fourier operations that map in and out of k-space
===================================================================================
"""

import torch

# %% forward Fourier
def Forward_Fourier(pixels):
    k_space = torch.rfft(pixels,2,onesided=False)
    return k_space

# %% inverse Fourier
def Inverse_Fourier(k_space):
    complex_pixels = torch.ifft(k_space,2)
    envelop_pixels = torch.sqrt(complex_pixels[:,:,:,:,0]**2 + complex_pixels[:,:,:,:,1]**2)
    return envelop_pixels

# %% full mapping with sampling mask
def Full_Map(pixels_in,mask):
    # make mask larger to account for complex part
    mask = mask.unsqueeze(4).repeat(1,1,1,1,2)
    
    # go through fourier space
    full_k_space = Forward_Fourier(pixels_in)
    sampled_k_space = full_k_space*mask
    pixels_out = Inverse_Fourier(sampled_k_space) 
    return pixels_out

# %% full mapping with sampling mask
def Full_Map_Pineda(pixels_in,mask):
    full_k_space = Forward_Fourier(pixels_in)
    sampled_k_space = full_k_space*mask
    pixels_out = torch.ifft(sampled_k_space,2)
    pixels_out = pixels_out.transpose(1,4).reshape(pixels_in.size(0),2,pixels_in.size(2),pixels_in.size(3))
    return pixels_out

