"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : presaveMNIST.py
                    File to load MNIST data and split into train,validate and test set    Author        : Iris Huijben
    Date          : 24/07/2019
    Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019

==============================================================================
"""

# File to load cifar10 data and split into train,validate and test set

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import os
from pathsetupMNIST import in_dir

num_classes = 10
# Load The data, split between train and test sets:
# The targets are the labels for classification
(x_train, y_train), (x_valANDtest, y_valANDtest) = mnist.load_data()

#%%
# Convert targets to one-hot vectors for classification.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valANDtest = keras.utils.to_categorical(y_valANDtest, num_classes)
#
## Split into separate validation and test set
x_val, x_test, y_val, y_test = train_test_split(x_valANDtest, y_valANDtest, test_size=0.5, random_state=1)
#%%
savedir = in_dir
np.save(os.path.join(savedir,'x_train.npy'),x_train)
np.save(os.path.join(savedir,'x_val.npy'),x_val)
np.save(os.path.join(savedir,'x_test.npy'),x_test)
np.save(os.path.join(savedir,'y_train.npy'),y_train)
np.save(os.path.join(savedir,'y_val.npy'),y_val)
np.save(os.path.join(savedir,'y_test.npy'),y_test)