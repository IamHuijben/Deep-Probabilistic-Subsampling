"""
===================================================================================
    Source Name   : prints.py
    Description   : function that prints intermediate losses and estimates time
                    until completion
===================================================================================
"""
# %% import dependencies
import numpy as np

# %% prints
def Intermediate(epoch,results,args,time_taken):
       train_acc = results['train acc'][-1]
       val_acc  = results['val acc'][-1]
       
       hours_taken = int(np.floor(time_taken/3600))
       time_inter = time_taken%3600 
       minutes_taken = int(np.floor(time_inter/60))
       seconds_taken = int(time_inter%60)

       epochs_left = args.no_epochs - epoch
           
       time_per_epoch = time_taken/epoch
       time_left = time_per_epoch*epochs_left
           
       hours_left = int(np.floor(time_left/3600))
       time_inter = time_left%3600 
       minutes_left = int(np.floor(time_inter/60))
       seconds_left = int(time_inter%60)
    
       print(f"| {epoch:>5} | {train_acc:>9.4} | {val_acc:>7.4} | {hours_taken:>2}h{minutes_taken:>3}m{seconds_taken:>3}s | {hours_left:>2}h{minutes_left:>3}m{seconds_left:>3}s |")
       
def Table():
       print("------------------------------------------------------------")
       print("| epoch | train acc | val acc |  time taken |   time left |")