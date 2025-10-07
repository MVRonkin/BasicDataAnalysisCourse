import numpy as np
import matplotlib.pyplot as plt 

from ._basic_tree import BasicTree

class DecisionTreeID3(BasicTree):
    
    #------------------------------------------
    def _best_threshold(self,X,y,feature):
    
        thresholds = np.unique(X[:,feature])
        gains      = np.zeros(thresholds.size)

        for i,threshold in enumerate(thresholds):                    
            y1, y2 = self._split_y(X,y, feature, threshold)                    

            if (y1.size>0) and (y2.size>0):
                gains[i] = self._info_gain(y,y1,y2)
            else:
                 gains[i] = 0

        idx_max = np.argmax(gains)            
        return thresholds[idx_max], gains[idx_max]
    
    #------------------------------------------
    def _best_split(self,X,y):
        samples, features = X.shape
        gains      = np.zeros(features)
        thresholds = np.zeros(features) 

        for feature in range(features):
            thresholds[feature], gains[feature] = self._best_threshold(X,y,feature)

        best_feature = np.argmax(gains)  # best index    

        return best_feature,thresholds[best_feature], gains[best_feature]  

    #------------------------------------------
    def _info_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        return self._entropy(y) - p * self._entropy(y1) - (1 - p) * self._entropy(y2)
    #------------------------------------------ 
    def _entropy(self,y):
        values,counts = np.unique(y, return_counts=True)
        p = counts/y.size
        return - np.sum(p*np.log2(p))
    
    #------------------------------------------
    def _to_result(self, y):
        return self._to_class(y)
    
    #------------------------------------------
    def _to_class(self, y):
        y = np.asarray(y)
        if y.ndim >0:
            values,counts = np.unique(y, return_counts=True)
            i_max = np.argmax(counts)
            return values[i_max]
        else:
            return y

    #---------------------------------
    def score(self, X, y):
        yhat  = self.predict(X)
        return sum((yhat==y)*1)/y.size