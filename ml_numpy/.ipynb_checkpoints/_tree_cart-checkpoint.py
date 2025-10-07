import numpy as np
import matplotlib.pyplot as plt 

from ._tree_id3 import DecisionTreeID3

class DecisionTreeСART(DecisionTreeID3):
    
    def _gini(self, y): #impurity
        n_samples = y.shape[0]
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum(np.square(counts/n_samples))

    #------------------------------------------
    def _info_gain(self, y, y1, y2): # impurity
        p = len(y1) / len(y)
        return self._gini(y) - (self._gini(y1) * p + self._gini(y2) * (1-p))

    #------------------------------------------
    def _predict(self, x, tree=None):
        if tree is None:
            tree = self.root
        
        if tree.is_leaf:
            return tree.value

        if x[tree.feature] >= tree.threshold:
            branch = tree.left
        else:
            branch = tree.right

        return self._predict(x, branch)   
    
    #------------------------------------------ 
    # Class method 
    def _split(self,X, feature, threshold):
        idx_r = np.where(X[:,feature] <  threshold)[0]
        idx_l = np.where(X[:,feature] >= threshold)[0]
        return X[idx_l,:], X[idx_r,:]
    #-----------------------------------------