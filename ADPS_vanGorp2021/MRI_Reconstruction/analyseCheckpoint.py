"""
===================================================================================
    Source Name   : analyseCheckpoint.py
    Description   : This file specifies the final testing procedure based on three
                    metrics: NMSE, PSNR, and SSIM
===================================================================================
"""
# %% import dependencies
import torch
import argparse

import loadData
import cudaDeviceChecker
import myModel
import testModel
       
# %% go through the test and print the results
if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-use_gpu',type =bool,help='If true will try to use cuda, otherwise will do on cpu',default=True)
    parser.add_argument('-gpu_index',type =int,help='index of cuda device to use when use_gpu=True',default=0)
    
    parser.add_argument('-batch_size',type=int,help='Batch size to use',default=12)
    parser.add_argument('-Pineda',type=bool,help='Fetch the bigger dataset following Pineda et al.',default=False)
    
    parser.add_argument('-sampling',type =str,help='choose from: "random_uniform", "low_pass", "VDS", "GMS", "LOUPE", "DPS", "ADPS"',default='DPS')
    parser.add_argument('-temperature',type =float,help='temperature scaling for softmax in DPS and ADPS',default=2.0)
    parser.add_argument('-no_lines',type =int,help='Number of lines to sample',default=26)
    parser.add_argument('-seed',type =str,help='which seed to checkpoint',default=0)
    
    args = parser.parse_args()
    
    # setup correct name
    args.save_name = args.sampling +"_"+str(args.no_lines)+"lines_"+str(args.seed)+"seed"
    
    if args.Pineda == True:
        args.save_name += "_Pineda"
    
    # set up the gpu
    args.device = cudaDeviceChecker.device(args)

    # put a limit on cpu resources used by pytorch
    torch.set_num_threads(8)
    torch.random.manual_seed(args.seed)
    
    # load data
    _,_,dataloader_test = loadData.load(args,only_test=True)
    
    # create network depending on the type of sampling performed
    if args.sampling in ["random_uniform", "low_pass", "VDS", "GMS"]:
        Network = myModel.fixedMask(args)
    if args.sampling == "LOUPE":
        Network = myModel.LOUPEMask(args)
    if args.sampling == "DPS":
        Network = myModel.DPSMask(args)
    if args.sampling == "ADPS" and args.Pineda == False:
        Network = myModel.ADPSMask(args)
    if args.sampling == "ADPS" and args.Pineda == True:
        Network = myModel.ADPSMask_legacy(args)
        
    Network = Network.to(args.device)
    
    # load checkpoint
    state_dict = torch.load("checkpoints/"+args.save_name+".tar",map_location=args.device)
    Network.load_state_dict(state_dict)
    
    # test network
    mse_mean,PSNR_mean,SSIM_mean = testModel.Test(Network,dataloader_test,args)