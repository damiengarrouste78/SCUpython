# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:46:25 2018

@author: dgr
"""
import numpy as np
def standardisation(x):     
    x-=np.mean(x) # the -= means can be read as x = x- np.mean(x)
    x/=np.std(x)
    return x