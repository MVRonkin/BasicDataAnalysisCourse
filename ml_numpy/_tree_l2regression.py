import numpy as np
import matplotlib.pyplot as plt 

from ._tree_cart import DecisionTreeСART

class RegressionTree(DecisionTreeСART):
    def _info_gain(self, y, y1, y2): # variance change
        p = y1.size / y.size
        return np.sum(np.var(y) - (p * np.var(y1) + (1-p) * np.var(y2)))
        
        
    #------------------------------------------
    def _to_result(self, y):
        return np.mean(y)
    
    #---------------------------------
    def score(self, X, y):
        yhat  = self.predict(X)
        return 1-  np.sum(np.square(yhat-y))/np.sum(np.square(y-y.mean()))