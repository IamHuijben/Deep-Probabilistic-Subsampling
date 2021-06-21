"""
===================================================================================
    Source Name   : cudaDeviceChecker.py
    Description   : Script used to check whether cuda is available on this machine
===================================================================================
"""
import torch

def device(args):
    #check for cpu
    if args.use_gpu == False:
        return "cpu"
    
    #check for some common errors
    if args.use_gpu == True and torch.cuda.is_available() == False:
        raise Exception("Tried using gpu, but cuda is not available.")
        
    if args.gpu_index >= torch.cuda.device_count():
        raise Exception(f"Tried using gpu at index {args.gpu_index}, which is not available.")
        
    # no errors found, return cuda device
    return torch.device(args.gpu_index)