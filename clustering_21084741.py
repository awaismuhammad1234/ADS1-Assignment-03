
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:39:56 2023

@author: ALIENWARE-CERDAS
"""
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
import numpy as np
import sklearn.cluster as cluster
from scipy.optimize import curve_fit
import pandas as pd
import errors as err
import scipy.optimize as opt

def logistics(t, a, k, t0):

  f = a / (1.0 + np.exp(-k * (t - t0)))
  return f
def poly(t, c0, c1, c2, c3):

  t = t - 1950
  f = c0 + c1*t + c2*t**2 + c3*t**3
  return f
