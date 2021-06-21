"""
===================================================================================
    Source Name   : TrainModel.py
    Description   : This file specifies a training and validation loop for MNIST
===================================================================================
"""
# %% import dependencies
import torch
import torch.nn.functional as F
import time
import numpy as np
import prints

# %% train and validate
def execute(Network,optimizer,train_loader,val_loader,args):
       # %% start the timer
       start_time = time.time()
       
       # %% initialize dictionary for resutls
       results = {}
       results['val loss']  = []
       results['val acc']   = []
       results['train loss'] = []
       results['train acc']  = []
       
       # %% run evaluation to get the starting point
       val_loss,val_acc = validate(Network,val_loader,args)
       
       # %% append results
       results['val loss'].append(val_loss.item())
       results['val acc'].append(val_acc)
       results['train loss'].append(val_loss.item())
       results['train acc'].append(val_acc)
       
       # %% do a print of a table
       prints.Table()
       
       # %% go over the epochs
       for epoch_id in range(args.no_epochs):
              # do the training
              train_loss,train_acc = train(Network,train_loader,args,optimizer)
              
              # do a validation
              val_loss,val_acc = validate(Network,val_loader,args)
              
              # append results
              results['val loss'].append(val_loss.item())
              results['val acc'].append(val_acc)
              results['train loss'].append(train_loss.item())
              results['train acc'].append(train_acc)
       
              # print results
              time_taken = time.time()-start_time
              prints.Intermediate(epoch_id+1,results,args,time_taken)
              
              # reprint table every 10 epochs
              if (epoch_id+1)%10 == 0 and (epoch_id+1)!=args.no_epochs:
                     prints.Table()
                     
       # check if current validation accuracy is the best
       state_dict = Network.state_dict()
       torch.save(state_dict,"checkpoints\\"+args.save_name+".tar")
                     
       return results

# %% validation function
def validate(Network,val_loader,args):
       # initialize some values
       loss = 0
       acc = 0
       
       # set Network to evaluation mode
       Network.eval()
       
       # go over the validation data
       for batch_id, (x,s_gt) in enumerate(val_loader):
              
              # input output relation
              x = x.to(args.device)
              with torch.no_grad():
                     s_hat_all = Network(x)
                     
              # calulcate acc
              acc += float(torch.sum(torch.argmax(s_hat_all[:,:,-1],dim=1) == s_gt.to(args.device)).item())/float(s_gt.size(0))
                     
              # create the ground truth with the same number of iterations
              s_gt_all = s_gt.to(args.device).unsqueeze(1).repeat(1,s_hat_all.size(2))
              
       
              # calculate the loss
              batch_loss = F.cross_entropy(s_hat_all,s_gt_all)
              loss += batch_loss
       
       loss /= (batch_id+1)
       acc /= (batch_id+1)
       
       return loss,acc

# %% training function
def train(Network,train_loader,args,Optimizer):
       # initialize some values
       loss = 0
       acc = 0
       
       # set Network to training mode
       Network.train()
       
       # go over the validation data
       for batch_id, (x,s_gt) in enumerate(train_loader):
              # put gradients back to 0
              Optimizer.zero_grad()
              
              # input output relation
              x = x.to(args.device)
              s_hat_all = Network(x)
              
              # calulcate acc
              acc += float(torch.sum(torch.argmax(s_hat_all[:,:,-1],dim=1) == s_gt.to(args.device)).item())/float(s_gt.size(0))
              
              # create the ground truth with the same number of iterations
              s_gt_all = s_gt.to(args.device).unsqueeze(1).repeat(1,s_hat_all.size(2))
              
       
              # calculate the loss
              batch_loss = F.cross_entropy(s_hat_all,s_gt_all)
              loss += batch_loss
              
              
              # backwards call and optimizer step
              batch_loss.backward()
              Optimizer.step()
       
       
       loss /= (batch_id+1)
       acc /= (batch_id+1)
       
       return loss,acc
       
# %% test function
def test(Network,test_loader,args):
       # initialize some values
       loss = 0
       acc = 0
       
       # set Network to evaluation mode
       Network.eval()
       
       # go over the validation data
       for batch_id, (x,s_gt) in enumerate(test_loader):
              
              # input output relation
              x = x.to(args.device)
              with torch.no_grad():
                     s_hat_all = Network(x)
                     
              # calulcate acc
              acc += float(torch.sum(torch.argmax(s_hat_all[:,:,-1],dim=1) == s_gt.to(args.device)).item())/float(s_gt.size(0))
                     
              # create the ground truth with the same number of iterations
              s_gt_all = s_gt.to(args.device).unsqueeze(1).repeat(1,s_hat_all.size(2))
              
       
              # calculate the loss
              batch_loss = F.cross_entropy(s_hat_all,s_gt_all)
              loss += batch_loss

       
       loss /= (batch_id+1)
       acc /= (batch_id+1)
       
       return loss,acc