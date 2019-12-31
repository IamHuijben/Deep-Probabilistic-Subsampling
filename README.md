# Deep Probabilistic Subsampling

Deep Probabilistic Subsampling allows learning representations of signals/images that are an exact subset of the original signal, i.e. subsampled signal.
The model jointly performs a task on the subsampled signal, e.g. reconstruction or classification. 


This repo provides the code for the three experiments presented in our paper:

Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun - [Deep probabilistic subsampling for task-adaptive compressed sensing](https://openreview.net/forum?id=SJeq9JBFvH)


## How to run the code

### Dependencies
TODO

### Experiments with MNIST (section 4.1) and CIFAR10 (section 4.3)
- Run pathsetupMNIST.py and pathsetuptCIFAR10.py respectively.

- Run presaveMNISt.py and presaveCIFAR10.py respectively.

- Run main_MNIST.py or main_CIFAR10.py to run one of the two experiments. 
These main files contain the settings as used in the paper.
These files train the corresponding model (found in myModels.py) and save it after training.
During training, training updates are plotted and saved (if savedir is not empty). 
When running from the terminal, make sure to uncomment plotting of the figures in the callbacks.

- Fill in the modelname (versionName parameter) and the saved weight file for which to run inference in Inference.py and run the file.
This file shows inference results on the test set and saves the results if savefigs = True.


### Experiment toy dataset, lines and circles (section 4.2)
- Run main.py to train the model. 
The parameters are set according to the values mentioned in the paper.
Data is generated in an on-line fashion, and after training the model weights are automatically saved.

- Fill in the modelname (versionName parameter) and the saved weight file for which to run inference in Inference.py and run the file.
This file shows inference results on a randomly pre-generated test set (testSet.npy and testSetY.npy) and saves the results if savefigs = True.
A pregenerated test set is used to fairly compare results among models. 

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{
huijben2020deep,
title={Deep probabilistic subsampling for task-adaptive compressed sensing},
author={Iris A.M. Huijben and Bastiaan S. Veeling and Ruud J.G. van Sloun},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SJeq9JBFvH}
}
```