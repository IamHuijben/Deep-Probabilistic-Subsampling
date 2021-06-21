"""
===================================================================================
    Source Name   : DataLoaders.py
    Description   : Function that creates the train and test data loaders
===================================================================================
"""
# %% import dependencies
import torch
import torchvision

# %% create dataloader
def load(args):
       batch_size = args.batch_size
       
       # train and validation loader
       MNIST_data = torchvision.datasets.MNIST('data/',train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                         ]))
       
       train_data, val_data = torch.utils.data.random_split(MNIST_data, [50000,10000])
       
       train_loader = torch.utils.data.DataLoader(
                     train_data,batch_size=batch_size,shuffle=True,
                     pin_memory = True, drop_last=True)
       
       val_loader = torch.utils.data.DataLoader(
                     train_data,batch_size=batch_size,shuffle=True,
                     pin_memory = True, drop_last=True)

       # test loader
       test_loader = torch.utils.data.DataLoader(
                     torchvision.datasets.MNIST('data/',train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                         ])),batch_size=batch_size,shuffle=True,
                                         pin_memory = True, drop_last=True)
       
       return train_loader,val_loader,test_loader
       