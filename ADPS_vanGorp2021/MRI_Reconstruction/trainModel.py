"""
===================================================================================
    Source Name   : TrainModel.py
    Description   : This file specifies a training and evaluation loop for MRI
===================================================================================
"""
# %% import dependencies
import torch
import torch.nn.functional as F
import time
import prints

# %% cropping to the central 320 by 320 pixels
def crop(image):   
    new_image = image[:,:,24:344,160:480]
    return new_image
    

# %% training
def train(Network, Discriminator, optimizer_recon, optimizer_disc, args, images_in):
    # set both models to training mode
    Network.train()
    Discriminator.train()
    
    # %% Discriminator
    # set the gradients to zero
    optimizer_recon.zero_grad()
    optimizer_disc.zero_grad()
    
    # input the images to reconnet without gradient for now
    with torch.no_grad():
        images_out = Network(images_in)
        
    if args.Pineda == True:
        images_in2 = crop(images_in)
        images_out = crop(images_out)
    else:
        images_in2 = images_in
        
    #check if we need to deal with ADPS
    if args.sampling == 'ADPS':
        multiplier = args.no_lines
        images_in2 = images_in2.repeat(1,multiplier,1,1)
        images_in2 = images_in2.reshape(args.batch_size*multiplier,1,images_in2.size(2),images_in2.size(3))
        images_out = images_out.reshape(args.batch_size*multiplier,1,images_out.size(2),images_out.size(3))
    else:
        multiplier = 1
    
    # Discriminator loss itself
    # create labels for the batch
    labels_gt = torch.zeros((2 * args.batch_size*multiplier,1)).to(args.device)
    labels_gt[0:args.batch_size*multiplier] = 1
    
    # append the images
    discriminator_in = torch.cat((images_in2,images_out),dim=0)
    
    # discriminate
    labels_out,features_out = Discriminator(discriminator_in)
    
    # get the loss
    loss_disc = F.binary_cross_entropy(labels_out,labels_gt)
    
    # call a backward on this loss and update the discriminator weights
    loss_disc.backward()
    optimizer_disc.step()
    
    # %% generator
    # put gradients back to zero
    optimizer_recon.zero_grad()
    optimizer_disc.zero_grad()
    
    # use the generator to output an image
    images_out = Network(images_in)
    
    if args.Pineda == True:
        images_out = crop(images_out)
    if args.sampling == 'ADPS':
        images_out = images_out.reshape(args.batch_size*multiplier,1,images_out.size(2),images_out.size(3))
    
    # find out what the discriminator thinks of this
    labels_out,features_out_new = Discriminator(images_out)
    
    # create new ground truth labels (aimed to trick the discriminator)
    labels_gt = torch.ones((args.batch_size*multiplier,1)).to(args.device)
    
    # create the ground truth features from the previous run
    features_gt = features_out[0:args.batch_size*multiplier].detach()
    
    # discrimantor to generator losses
    loss_recon_disc = F.binary_cross_entropy(labels_out.squeeze(),labels_gt.squeeze())
    loss_recon_feat = F.mse_loss(features_out_new.reshape(64*args.batch_size*multiplier),features_gt.reshape(64*args.batch_size*multiplier))
    
    # get the mse loss between the images
    loss_recon_mse = F.mse_loss(images_in2,images_out)
    
    # add the losses
    loss_recon = args.weight_mse*loss_recon_mse + args.weight_disc*loss_recon_disc + args.weight_disc_features*loss_recon_feat
    
    # do a backward call on the generator
    loss_recon.backward()
    optimizer_recon.step()

    return loss_recon.item(), loss_disc.item()
    
# %% evaluate
def evaluate(Network, Discriminator, args, dataloader_val):
    # set both models to eval mode
    Network.eval()
    Discriminator.eval()
    
    val_loss_recon_average = 0
    val_loss_disc_average = 0
    
    # loop over all validation data
    for batch_id,images_in in enumerate(dataloader_val):
        images_in = images_in.to(args.device)

        # no gradient required
        with torch.no_grad():
            # input the images to reconnet
            images_out = Network(images_in)
            
            #crop the images if necesary
            if args.Pineda == True:
                images_in = crop(images_in)
                images_out = crop(images_out)
            
            #check if we need to deal with ADPS
            if args.sampling == 'ADPS':
                multiplier = args.no_lines
                images_in = images_in.repeat(1,multiplier,1,1)
                images_in = images_in.reshape(args.batch_size*multiplier,1,images_in.size(2),images_in.size(3))
                images_out = images_out.reshape(args.batch_size*multiplier,1,images_out.size(2),images_out.size(3))
            else:
                multiplier = 1
                
            
            # input to discriminator
            discriminator_in = torch.cat((images_in,images_out),dim=0)
            labels_out,features_out = Discriminator(discriminator_in)
            
            # ground truth labels
            labels_gt = torch.zeros((2 * args.batch_size*multiplier,1)).to(args.device)
            labels_gt[0:args.batch_size*multiplier] = 1
            
            # split the labels for the reconstrcution loss
            labels_gt_recon = torch.ones((args.batch_size*multiplier,1)).to(args.device)
            labels_recon = labels_out[args.batch_size*multiplier:2*args.batch_size*multiplier]
            
            # split the features
            features_gt   = features_out[0:args.batch_size*multiplier]
            features_fake = features_out[args.batch_size*multiplier:2*args.batch_size*multiplier]
            
            # get the mse loss between the images
            loss_recon_mse = F.mse_loss(images_in,images_out)
        
            # losses
            loss_recon_disc = F.binary_cross_entropy(labels_recon,labels_gt_recon)
            loss_recon_feat = F.mse_loss(features_gt,features_fake)
            loss_disc = F.binary_cross_entropy(labels_out,labels_gt)
        
            # add the losses
            loss_recon = args.weight_mse*loss_recon_mse + args.weight_disc*loss_recon_disc + args.weight_disc_features*loss_recon_feat

        val_loss_recon_average += loss_recon
        val_loss_disc_average  += loss_disc
        
        break

        
    # average the results
    val_loss_recon_average /= (batch_id+1)
    val_loss_disc_average  /= (batch_id+1)
    
    return val_loss_recon_average.item(),val_loss_disc_average.item()


# %% define the training and validation loops
def execute(Network, Discriminator, optimizer_recon, optimizer_disc, args, dataloader_train,dataloader_val):
    print("started training\n")
    # start the timer
    start_time = time.time()
    
    # %% initialize dictionary for results
    results = {}
    
    # %% perform an initial validation
    val_loss_recon_average,val_loss_disc_average = evaluate(Network, Discriminator, args, dataloader_val)
    
    results['val_loss_recon']  = [val_loss_recon_average]
    results['val_loss_disc']   = [val_loss_disc_average]
    results['train_loss_recon']  = [val_loss_recon_average]
    results['train_loss_disc']   = [val_loss_disc_average]
    
    # %% loop over all the epochs
    for epoch_id in range(args.no_epochs):
        # print for this epoch
        prints.print_epoch()
        
        # %% training
        #   loop over all the data in this epoch
        for batch_id,images_in in enumerate(dataloader_train):
            images_in = images_in.to(args.device)
            
            # Train!
            train_loss_recon, train_loss_disc = train(Network, Discriminator, optimizer_recon, optimizer_disc, args, images_in)
            
            # append the results
            results['train_loss_recon'].append(train_loss_recon)
            results['train_loss_disc'].append(train_loss_disc)
            
            # print
            counter = 200
            
            if (batch_id+1) % counter == 0:
                prints.print_intermediate(epoch_id,batch_id,start_time,args,results,dataloader_train)
                
                state_dict = Network.state_dict()
                torch.save(state_dict,"checkpoints\\"+args.save_name+"_"+str(epoch_id)+".tar")
                
        # %% validation
        val_loss_recon_average,val_loss_disc_average = evaluate(Network, Discriminator, args, dataloader_val)
        
        results['val_loss_recon'].append(val_loss_recon_average)
        results['val_loss_disc'].append(val_loss_disc_average)
        
        # saving
        state_dict = Network.state_dict()
        torch.save(state_dict,"checkpoints\\"+args.save_name+".tar")
        
    # %% return the results
    return results