"""
===================================================================================
    Source Name   : fixedMasks.py
    Description   : Script used to create the various sampling masks that are not
                    changed once created.
===================================================================================
"""

import torch

def createMask(args):
    if args.Pineda == True:
        width = 368
        height = 640
    else:
        width = 208
        height = 208
    
    # initialize the sample matrix A
    A = torch.zeros(width,height)
    
    # %% random_uniform
    if args.sampling == "random_uniform":
        # create a random permutiation of the 208 lines
        rand_line_indices = torch.randperm(width)
        
        # select only the number of lines needed
        indices_to_sample = rand_line_indices[0:args.no_lines]
        
        # fill in those lines with ones 
        A[indices_to_sample,] = 1

    # %% low_pass
    elif args.sampling == "low_pass":
        # find lowest and highest line that need to be sampled
        min_line_index = int(width/2 - args.no_lines/2)
        max_line_index = int(width/2 + args.no_lines/2)
        
        # fill in those lines with ones
        A[min_line_index:max_line_index,] = 1

    # %% VDS
    elif args.sampling == "VDS":
        # create tensor that indexes all the lines
        line_indices = torch.tensor(range(width)).type(torch.float32)
        
        # create a polynomial distribution
        half = (width-1)/2
        distribution = (1 - abs(line_indices-half)/half)**6
        
        # sample from this distribution using gumbel noise (GN)
        logits = torch.log(distribution/(1-distribution+1e-20))
        GN = -torch.log(-torch.log(torch.rand(width,) +1e-20)+1e-20)
        
        _,indices_to_sample = torch.topk(logits+GN,args.no_lines)
        
        # fill in those lines with ones 
        A[indices_to_sample,] = 1

    # %% GMS
    elif args.sampling == "GMS":
        if args.no_lines>26:
            raise Exception('Cannot use GMS for more than 26 lines because they are not available.')
        if args.Pineda == True:
            raise Exception('Cannot use GMS for the regime of Pineda because lines are not available.')
        indices_to_sample2 = torch.tensor([103,99,107,104,102,105,101,108,98,110,112,100,106,115,92,94,113,96,109,126,122,91,82,120,117,111])
        indices_to_sample = indices_to_sample2[0:args.no_lines]
        A[indices_to_sample,] = 1
        
    # %% return A
    # roll A to make sure it complies with the definition of the fourier transform
    half = int(width/2)
    A = torch.roll(A,shifts =(-half),dims = (0))
    
    # expand allong batch dimension
    A = A.reshape(1,1,width,height).repeat(args.batch_size,1,1,1).to(args.device)
    
    return A
