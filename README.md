This repository provides the code for the following two papers:

Iris A.M. Huijben*, Bastiaan S. Veeling*, and Ruud J.G. van Sloun - [Deep probabilistic subsampling for task-adaptive compressed sensing](https://openreview.net/forum?id=SJeq9JBFvH)
\* equal contribution

Hans van Gorp, Iris A.M. Huijben, Bastiaan S. Veeling, Nicolla Pezzotti, and Ruud J.G. van Sloun - [Active Deep probabilistic Subsampling]

Code for both papers is split in two separate folders (DPS_Huijben2020 and ADPS_vanGorp20201), for which the documentation can be found below

# Deep Probabilistic Subsampling (DPS)

Deep Probabilistic Subsampling allows learning representations of signals/images that are an exact subset of the original signal, i.e. subsampled signal.
The model jointly performs a task on the subsampled signal, e.g. reconstruction or classification. 

## How to run the code

All code can be found in the folder DPS_Huijben2020. The full code was written in tensorflow, so the following instructions belong to the tensorflow code.
However, recently we added a Pytorch folder with the DPS-topk implementation in pytorch. This is not a full model implementation, but should facilitate Pytorch users with implementation the full model in Pytorch. 

### Dependencies
Download the [anaconda](https://www.anaconda.com/) package. 

In the anaconda prompt run:
```
conda env create -f DPS_environment.yml
```
and then activate the environment:
```
conda activate DPS
```

### Experiments with MNIST (section 4.1) and CIFAR10 (section 4.3)
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

## Citation

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


# A-DPS

Active Deep Probabilistic Subsampling learns to subsample incoming data, conditioned on already sampled data, making it an extension of DPS.

## Directory Structure

All code for this paper can be found in the ADPS_vanGorp2021 folder.

- 'ADPS_environment.yml': Environment file for anaconda that contains all dependencies to run the code in PyTorch.
- 'MNIST Classification': Contains the MNIST [1] classification experiment. Here, we compare DPS with A-DPS at different subsampling ratios.
- 'MRI Reconstruction': Contains the MRI reconstruction experiment for the fastMRI knee data [2].

## Environment
To make use of the environment, first install [Anaconda](https://www.anaconda.com/). In the anaconda prompt run:
```
conda env create -f ADPS_environment.yml
```
and then activate the environment:
```
conda activate ADPS_env
```

## MNIST Classification
To run the MNIST Classification example run the following command: 
```
MNIST_Classification/main.py -sampling 'ADPS' -percentage 4
```
It is also possible to sample using 'DPS' or at a different percentage.
To analyze this checkpoints run:
```
MNIST_Classification/analyseCheckpoint.py -sampling 'ADPS' -percentage 4
```

## MRI Reconstruction
### Data
To run the MRI reconstruction it is first required to download the single coil knee training and validation data from the [fastMRI website](https://fastmri.med.nyu.edu). To perform the preprocessing run:
```
MRI_Reconstruction/preprocessing.py -path "mypath"
```
where mypath points towards the downloaded dataset.

### Experiment
To then run the MRI experiment run the following command:
```
MRI_Reconstruction/main.py -sampling 'ADPS'
```
It is possible to choose from one of the following sampling strategies:
- 'random uniform', this will results in samples drawn from a uniform distribution
- 'low pass', this will results in only the samples closest to the DC frequency to be used.
- 'VDS', Variable Density Sampling results in samples drawn from a polynomial distribution with the highest probability for the DC line, and lower probability for higher frequency lines.
- 'GMS', Greedy Mask Selection from Sanchez et al. [3]. Note that we use pretrained lines here. To run their code, see their [github](https://github.com/t-sanchez/stochasticGreedyMRI).
- 'LOUPE', Learning-based Optimization of the Under-sampling PattErn from Bahadir et al. [4]. Note that we have reimplemented their code in pytorch, for the original version see their [github](https://github.com/cagladbahadir/LOUPE/).
- 'DPS', Deep Probabilistic Subsampling, as specified by Huijben et al. [5]
- 'ADPS', Active Deep Probabilistic Subsampling, the active extension of DPS.

In a similar way, a saved checkpoint can be analyzed:
```
MRI_Reconstruction/analyseCheckpoint.py -sampling 'ADPS'
```


## References
[1] Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278–2323, 1998.

[2] Jure Zbontar, et al. "fastMRI: An open dataset and benchmarks for accelerated MRI." arXiv preprint arXiv:1811.08839 (2018).

[3] Thomas Sanchez, Baran Gozcu, Ruud B. van Heeswijk, Armin Eftekhari, Efe Ilicak, Tolga Cukur, and Volkan Cevher. “Scalable Learning-Based Sampling Optimization for Compressive Dynamic MRI,” in IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2020, Barcelona, Spain, 4 2020, pp. 8584–8588.

[4] Cagla D. Bahadir, Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. "Deep-learning-based optimization of the under-sampling pattern in mri." IEEE Transactions on Computational Imaging, 6:1139–1152, 2020.

[5] Iris A. M. Huijben, Bastiaan S. Veeling, and Ruud J. G. Van Sloun. “Deep Probabilistic Subsampling for Task-Adaptive Compressed Sensing,” in International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, 2020.

## Citation

Please cite our paper if you use this code in your own work:

```
@inproceedings{
vanGorp2021,
title={Active Deep probabilistic Subsampling},
author={H. van Gorp and Iris A.M. Huijben and Bastiaan S. Veeling and Nicola Pezzotti and Ruud J.G. van Sloun},
booktitle={International Conference on Machine Learning},
year={2021},
}
```
