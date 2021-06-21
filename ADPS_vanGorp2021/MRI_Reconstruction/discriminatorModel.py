"""
===================================================================================
    Source Name   : DiscriminatorModel.py
    Description   : Discriminator model used to promote visually plausible images
===================================================================================
"""
# %% import dependencies
import torch.nn as nn

# %% Discriminator Model
class Discriminator(nn.Module):
       def __init__(self,args):
           super(Discriminator,self).__init__()
           self.device = args.device
           
           self.disc_size = 64
           
           self.convolutions = nn.Sequential(
                   nn.Conv2d(1,self.disc_size,3,stride = 2, padding = 2),
                   nn.LeakyReLU(negative_slope = 0.2),
                   nn.Conv2d(self.disc_size,self.disc_size,3,stride = 2, padding = 2),
                   nn.LeakyReLU(negative_slope = 0.2),
                   nn.Conv2d(self.disc_size,self.disc_size,3,stride = 2, padding = 2),
                   nn.LeakyReLU(negative_slope = 0.2),
                   )

           self.pooling = nn.AdaptiveAvgPool2d(1)
           
           self.fc = nn.Sequential(
                   nn.Dropout(p=0.4),
                   nn.Linear(self.disc_size,1),
                   nn.Sigmoid(),
                   )
           
       # %% forward call
       def forward(self,input_images):
           output_conv = self.convolutions(input_images)
           features = self.pooling(output_conv).squeeze()
           labels = self.fc(features)
           return labels,features