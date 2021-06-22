"""
===================================================================================
    Source Name   : TestModel.py
    Description   : This file specifies the final testing procedure based on three
                    metrics: NMSE, PSNR, and SSIM
===================================================================================
"""
# %% import dependencies
import torch
import numpy as np
import skimage.metrics
import sys

# %% test
def crop(image):   
    new_image = image[:,:,24:344,160:480]
    return new_image
    

def Test(Network,dataloader_test,args):
       print("Started testing")
       # set to evaluation mode
       if args.sampling == 'LOUPE':
            Network.loupe.hard = True
       Network.eval()
       
       # get some dataset values
       dataset_size = dataloader_test.dataset.size(0)
       dataset_numbatches = np.ceil(dataset_size/args.batch_size)
       
       # presave the three metrics
       mse_all  = torch.zeros(dataset_size,)
       SSIM_all = torch.zeros(dataset_size)
       ratio_all = torch.zeros(dataset_size)
       
       # loop over all the test data
       for batch_id,images_in in enumerate(dataloader_test):
              # progress bar
              divisor = int(dataset_numbatches/100)
              if divisor == 0:
                  divisor = 1
              
              if batch_id%divisor==0:
                  sys.stdout.write('\r')
                  j = (batch_id + 1) / dataset_numbatches
                  sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
                  sys.stdout.flush()
              
              # push images to cuda
              images_in = images_in.to(args.device)
              
              # do the reconstruction
              with torch.no_grad():
                     images_out = Network(images_in)
              images_out = images_out[:,-1,:,:].unsqueeze(1)
                     
              # crop if necesarry
              if args.Pineda == True:
                  images_in = crop(images_in)
                  images_out = crop(images_out)
                     
              # get the mse for this batch
              mse = (((images_in-images_out)**2).mean(dim=(1,2,3)))
        
              n_mse = ((((images_in-images_out)**2).sum(dim=(1,2,3)))/((images_in**2).sum(dim=(1,2,3))))
        
              mse_all[batch_id*args.batch_size:(batch_id+1)*args.batch_size]  = n_mse
        
              # Get the PSNR ratio for this batch
              maximum_pixel,_ = torch.max(images_in,dim = 2)
              maximum_pixel,_ = torch.max(maximum_pixel,dim = 2)
              maximum_pixel = maximum_pixel.squeeze()
        
              ratio = maximum_pixel**2/mse
              
              ratio_all[batch_id*args.batch_size:(batch_id+1)*args.batch_size] = ratio

              # get the SSIM for this batch
              for j in range(args.batch_size):
                  SSIM_all[batch_id*args.batch_size+j] = skimage.metrics.structural_similarity(
                          images_in[j,0,:,:].cpu().numpy(),
                          images_out[j,0,:,:].cpu().numpy(),
                          data_range=images_in[j,0,:,:].cpu().numpy().max(),
                          )
                                
       # %% calulcate the average results
       mse_mean  = torch.mean(mse_all)
       ratio_mean = torch.mean(ratio_all)
       PSNR_mean = 10 *torch.log10(ratio_mean)
       SSIM_mean = torch.mean(SSIM_all)
       
       # %% complete progress bar
       sys.stdout.write('\r')
       j = 1
       sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
       sys.stdout.flush()
       print('\n')
       
       # %% print
       print('Results on the test set:')
       print("----------------")
       print(f"| NMSE | {mse_mean:>6.3} |")
       print(f"| PSNR | {PSNR_mean:>6.3} |")
       print(f"| SSIM | {SSIM_mean:>6.3} |")
      
       # %% return results
       return mse_mean,PSNR_mean,SSIM_mean