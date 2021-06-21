"""
===================================================================================
    Source Name   : analyseCheckpoint.py
    Description   : This file specifies the final testing procedure
===================================================================================
"""
# %% import dependencies
import torch
import argparse

import loadData
import cudaDeviceChecker
import myModel
import trainModel
       
# %% go through the test and print the results
if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-use_gpu',type =bool,help='If true will try to use cuda, otherwise will do on cpu',default=True)
    parser.add_argument('-gpu_index',type =int,help='index of cuda device to use when use_gpu=True',default=0)
    
    parser.add_argument('-batch_size',type=int,help='Batch size to use',default=256)
    
    parser.add_argument('-sampling',type =str,help='choose from: "DPS" or "ADPS"',default='ADPS')
    parser.add_argument('-percentage',type =int,help='percentage of samples to use',default=1) 
    
    parser.add_argument('-temperature',type =float,help='temperature scaling for softmax in DPS and ADPS',default=2.0)
    parser.add_argument('-seed',type =str,help='which seed to use',default=0)
    
    parser.add_argument('-learning_rate',type =float,help='learning rate for network parameters',default=2e-4)
    parser.add_argument('-learning_rate_logits',type =float,help='learning rate for logit parameters for DPS and LOUPE',default=2e-3)
    parser.add_argument('-no_epochs',type =int,help='number of epochs to train for',default=2)
    
    
    
    args = parser.parse_args()
    
    args.no_pixels = int(28**2*args.percentage/100)
    
    # setup correct name
    args.save_name = args.sampling +"_"+str(args.percentage)+"percent_"+str(args.seed)+"seed"
    
    # set up the gpu
    args.device = cudaDeviceChecker.device(args)

    # put a limit on cpu resources used by pytorch
    torch.set_num_threads(8)
    torch.random.manual_seed(args.seed)
    
    # load data
    train_loader,val_loader,test_loader = loadData.load(args)
    
    # %% create network depending on the type of sampling performed
    Network = myModel.Network(args)
    Network = Network.to(args.device)
 
    # load the statedict
    statedict = torch.load("checkpoints/"+args.save_name+".tar",map_location=args.device)
    Network.load_state_dict(statedict)
    
    # %% test network
    loss,acc = trainModel.test(Network,test_loader,args)
    print(f"\n\n\n Accuracy on the test set = {100*acc:>.3}%")