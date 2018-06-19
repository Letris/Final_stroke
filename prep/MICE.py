from fancyimpute import MICE
import numpy as np

def MICE_impute(X):
"""impute missing values in X using the MICE algorithm. Input is a list of lists"""
    mice = MICE()
    X_imputed = mice.complete(np.array(X))
    print X_imputed
