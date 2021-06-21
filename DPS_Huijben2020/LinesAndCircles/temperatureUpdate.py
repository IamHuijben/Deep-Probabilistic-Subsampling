"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : temperatureUpdate.py
                    File contains both the tensorflow and python implementation 
                    of the function that returns the temperature parameter (dependent on epoch)
                    of the softmax function for soft-sampling / Gumbel-softmax sampling.
                    
    Author        : Iris Huijben
    Date          : 23/04/2019
     Reference     : Iris A.M. Huijben, Bastiaan S. Veeling, and Ruud J.G. van Sloun,
                    "Deep probabilistic subsampling for task-adaptive compressed sensing", 2019

==============================================================================
"""

import tensorflow as tf 

TempUpdateBasisTemp = 1.0
TempUpdateFinalTemp = 1.0

def temperature_update_tf(tempIncr, epoch, n_epochs):
    TempUpdate = (TempUpdateBasisTemp-TempUpdateFinalTemp)/(n_epochs-1)    
    return (tf.subtract(tf.to_float(TempUpdateBasisTemp),tf.multiply(tf.to_float(epoch),TempUpdate)))*tempIncr


def temperature_update_numeric(tempIncr, epoch, n_epochs): 
    TempUpdateTempUpdate = (TempUpdateBasisTemp-TempUpdateFinalTemp)/(n_epochs-1)    
    return (TempUpdateBasisTemp - (epoch) * TempUpdateTempUpdate)*tempIncr
