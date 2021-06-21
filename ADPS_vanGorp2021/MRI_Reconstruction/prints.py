# -*- coding: utf-8 -*-
"""
===================================================================================
    Source Name   : prints.py
    Description   : function that prints intermediate losses and estimates time
                    until completion
===================================================================================
"""
import time
import numpy as np

# %% printing of results
def print_intermediate(epoch_id,batch_id,start_time,args,results,dataloader_train):
    dataset_size = dataloader_train.dataset.size(0)
    dataset_numbatches = np.ceil(dataset_size/args.batch_size)
    
    examples_seen = (batch_id+1)*args.batch_size
    percentage = int(100 * examples_seen / dataset_size)
    
    epoch = epoch_id+1
    time_taken = time.time()-start_time
    
    train_loss = results['train_loss_recon'][-1]
    val_loss = results['val_loss_recon'][-1]
    
    hours_taken = int(np.floor(time_taken/3600))
    time_inter = time_taken%3600 
    minutes_taken = int(np.floor(time_inter/60))
    seconds_taken = int(time_inter%60)
    
    
    epochs_left = args.no_epochs - epoch
    batches_left_here = dataset_numbatches-batch_id -1
    total_no_batches_left = epochs_left*dataset_numbatches+ batches_left_here
    
    total_no_batches_seen = batch_id+1+epoch_id*dataset_numbatches
    
    time_per_batch = time_taken/total_no_batches_seen
    time_left = time_per_batch*total_no_batches_left
    
    hours_left = int(np.floor(time_left/3600))
    time_inter = time_left%3600 
    minutes_left = int(np.floor(time_inter/60))
    seconds_left = int(time_inter%60) 
    
    
    print(f"| {epoch:>5} | {percentage:>4}% | {train_loss:>10.4} | {val_loss:>9.4} | {hours_taken:>2}h{minutes_taken:>3}m{seconds_taken:>3}s | {hours_left:>2}h{minutes_left:>3}m{seconds_left:>3}s |")
       
def print_epoch():
    print("----------------------------------------------------------------------")
    print("| epoch | perc. | train loss |  val loss |  time taken |   time left |")
