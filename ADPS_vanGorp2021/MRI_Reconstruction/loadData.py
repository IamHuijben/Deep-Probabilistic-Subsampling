"""
===================================================================================
    Source Name   : LoadData.py
    Description   : Function used to load the preprocessed data
===================================================================================
"""
# %% import dependancies
import h5py
import os
import torch
import torch.utils.data as torchdata

# %% load the data sets
def fetch_data(dataset,in_dir):
     print('Loading ', dataset, '...')
     full_file = os.path.join(in_dir, dataset)
     data = h5py.File(full_file+'.h5', 'r')
     data = torch.tensor(data[list(data.keys())[0]][()]).unsqueeze(1)
     return data

def load(args,only_test=False):
    if args.Pineda == True:
        in_dir = 'data//preprocessed_Pineda'
    else:
        in_dir = 'data//preprocessed'
    
    if only_test == False:
        datasets = ['x_train', 'x_val', 'x_test']
    else: 
        datasets = ['x_test']
    
    dataloader_train = 0
    dataloader_val = 0
    
    for dataset in datasets:
        data = fetch_data(dataset,in_dir)
        
        if dataset == 'x_train':
            dataloader_train = torchdata.DataLoader(data, batch_size=args.batch_size,
                                                    pin_memory=True,shuffle = True,drop_last=True)
        if dataset == 'x_val':
            dataloader_val = torchdata.DataLoader(data, batch_size=args.batch_size,
                                                    pin_memory=True,shuffle = False,drop_last=True)
        if dataset == 'x_test':
            dataloader_test = torchdata.DataLoader(data, batch_size=args.batch_size,
                                                    pin_memory=True,shuffle = False,drop_last=True)
        
        
        
    print("finished loading data\n")
    return dataloader_train,dataloader_val,dataloader_test



