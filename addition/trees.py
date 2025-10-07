import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns; sns.set()
import os

class Node():
    def __init__(self, 
                 feature=None, 
                 threshold=None,
                 value=None,
                 samples = None,
                 probability = None,
                 left=None, 
                 right=None,
                 depth = None,
                 gain  = None,
                 is_leaf = False):
        
        self.feature    = feature   # Index of the feature (column)
        self.threshold  = threshold # Threshold value for feature (raw)
        self.value   = value        # Value if the node is a leaf in the tree
        self.samples = samples      # Number of samples in branch        
        self.depth   = depth       # Depth of the branch
        self.gain    = gain        # Gain in the Branch 
        self.left    = left        # 'true branch' subtree
        self.right   = right       # 'false branch' subtree        
        self.is_leaf = is_leaf     # if the node is a leaf or not 
        self.probability = probability # Probability of classes in the branch (for classification)
#------------------------------------------        
class BasicTree:
    def __init__(self, 
                 min_samples_split=2,
                 min_samples_leaf = 1,
                 min_gain=1e-7,
                 max_depth=-1):
        
        self.root = None  

        self.min_samples_split = min_samples_split
        
        self.min_samples_leaf = min_samples_leaf
        
        self.min_gain = min_gain

        self.max_depth = max_depth

        if max_depth<1:
            max_depth = float("inf")

        if self.min_samples_leaf>=self.min_samples_split:
            self.min_samples_split = self.min_samples_leaf +1
        
        
    #-----------------------------------------------
    # Class method 
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.root = Node(depth = 1)
        self.root = self._build(X, y)
        return self
    
    #-----------------------------------------------
    # Class method 
    def _build(self, X, y, depth=1):
        
        max_gain = 1
        best_feature   = 0
        best_threshold = 0
        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and depth <= self.max_depth:

            best_feature, best_threshold, max_gain= self._best_split(X,y)
            
            if max_gain > self.min_gain:

                X_l,y_l,X_r,y_r = self._split_xy(X,y, best_feature, best_threshold)
                
                return Node(feature=best_feature, 
                            threshold=best_threshold, 
                            left    = self._build(X_l,y_l, depth + 1), #true_branch, 
                            right   = self._build(X_r,y_r, depth + 1), #false_branch
                            depth   = depth,
                            samples = y.size,
                            gain    = max_gain,
                            probability = self._probabilities(y))

        
        leaf_value = self._to_result(y)
        p = self._probabilities(y)
        return Node(value   = leaf_value, 
                    is_leaf = True, 
                    depth   = depth,
                    samples = y.size,
                    gain    = max_gain,
                    probability = p)
    #------------------------------------------
    def _probabilities(self,y):
        values,counts = np.unique(y, return_counts=True)
        return counts/y.size
    #------------------------------------------
    # Generic method
    def _best_split(self,X,y):
        best_feature, best_threshold, max_gain = 0,0,1
        return best_feature, best_threshold, max_gain
    
    #------------------------------------------
    def _to_result(self, y):
        return 0
    #------------------------------------------
    def _predict(self, x, tree=None):
        if tree is None:
            tree = self.root
        
        if tree.is_leaf:
            return tree.value

        if x[tree.feature] == tree.threshold:
            branch = tree.left
        else:
            branch = tree.right

        return self._predict(x, branch)
    
    #------------------------------------------
    def predict(self, X):
        y_pred = [self._predict(sample) for sample in X]
        return np.asarray(y_pred)
    #------------------------------------------ 
    # Class method    
    def print_tree(self, indent="   " ):
        self._print_tree(indent=indent)
    
    #------------------------------------------ 
    # Class private method    
    def _print_tree(self, node=None, indent="   " ):
        if not node:
            node = self.root
        if node.is_leaf:
            print (node.value, node.probability, end = "\n")
        else:
            # Print test
            print ("%s:%.3f?" % (node.feature, node.threshold), end = "\n")
            # Print the true scenario
            print ("%sBRANCH = %s_LEFT (%d samples)->" % (indent, node.depth, node.samples), end=" ")
            self._print_tree(node.left, indent + indent)
            # Print the false scenario
            print ("%sBRANCH = %s_RIGHT (%d samples)->" % (indent, node.depth, node.samples), end=" ")
            self._print_tree(node.right, indent + indent)

    #------------------------------------------ 
    # Class method 
    def _split(self,X, feature, threshold):
        idx_r = np.where(X[:,feature] != threshold)[0]
        idx_l = np.where(X[:,feature] == threshold)[0]
        return X[idx_l,:], X[idx_r,:]
    
    #------------------------------------------ 
    # Class method 
    def _split_xy(self,X,y,feature, threshold):
        Xy = np.column_stack((X,y))
        Xy_l, Xy_r = self._split(Xy, feature, threshold)
        y_l = Xy_l[:,-1]
        y_r = Xy_r[:,-1]
        return Xy_l[:,:-1], y_l, Xy_r[:,:-1], y_r

    #------------------------------------------ 
    def _split_y(self,X,y,feature, threshold):
        Xy_l, Xy_r = self._split(np.column_stack((X,y)), feature, threshold)
        y_l = Xy_l[:,-1]
        y_r = Xy_r[:,-1]
        return y_l, y_r 
#-----------------------------
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
#----------------------------------------
class DecisionTreeС45(DecisionTreeID3):
    def _info_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        G = self._entropy(y) - p * self._entropy(y1) - (1 - p) * self._entropy(y2)
        return G/-(p * np.log2(p) + (1-p) * np.log2(1-p)) 
#--------------------------------------------
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

#-----------------------------------------    
class LogLossWithLogits:
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def activation(self,x):
        return self.sigmoid(x)

    def loss(self,y, y_hat):
        y_pred = np.clip(y_hat, 1e-15, 1 - 1e-15)
        p = self.sigmoid(y_hat)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    def gradient(self,y, y_hat):
        p = self.sigmoid(y_hat)
        return -(y - p)

    def hessian(self,y, y_hat):
        p = self.sigmoid(y_hat)
        return p * (1 - p)
    
#--------------------------
class BasicXGBTree(BasicTree):
    def __init__(self, 
                 min_samples_split=2,
                 min_samples_leaf = 1,
                 min_gain=1e-7,
                 max_depth=-1,
                 lambda_=0.0, 
                 gamma=0.0, 
                 loss_func = None):
        
        self.lambda_ = lambda_
        self.gamma   = gamma
        self.loss_func = loss_func
        
        if self.loss_func is None:
            self.loss_func = LogLossWithLogits()
        
        super(BasicXGBTree,self).__init__(min_samples_split,
                                          min_samples_leaf,
                                          min_gain,
                                          max_depth)
        
    #-----------------------------------------------
    # Class method 
    def fit(self, X, y, y_hat):
        self.n_samples, self.n_features = X.shape
        self.root = Node(depth = 1)
        self.root = self._build(X, y, y_hat)
        return self
    
    #-----------------------------------------------
    # Class method 
    def _build(self, X, y, y_hat, depth=1):
        max_gain = 1
        best_feature   = 0
        best_threshold = 0
        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and depth <= self.max_depth:

            best_feature, best_threshold, max_gain= self._best_split(X,y, y_hat)
            
            if max_gain > self.min_gain:

                X_l,y_l,y_hat_l,X_r,y_r,y_hat_r = self._split_xy(X,y, y_hat, best_feature, best_threshold)
                
                return Node(feature=best_feature, 
                            threshold=best_threshold, 
                            left    = self._build(X_l,y_l, y_hat_l, depth + 1), #true_branch, 
                            right   = self._build(X_r,y_r, y_hat_r, depth + 1), #false_branch
                            depth   = depth,
                            samples = y.size,
                            gain    = max_gain,
                            probability = self._probabilities(y))

        
        leaf_value = self._leaf_weight(y, y_hat) 
        p = self._probabilities(y)
        return Node(value   = leaf_value, 
                    is_leaf = True, 
                    depth   = depth,
                    samples = y.size,
                    gain    = max_gain,
                    probability = p)

    #------------------------------------------
    # Generic method
    def _best_split(self,X, y, y_hat):
        best_feature, best_threshold, max_gain = 0,0,1
        return best_feature, best_threshold, max_gain

    #------------------------------------------
    def _leaf_weight(self, y, y_hat):
        return self._to_result(y)   

    def _split(self, X, feature, threshold):
        idx_r = np.where(X[:,feature] <  threshold)[0]
        idx_l = np.where(X[:,feature] >= threshold)[0]
        return X[idx_l,:], X[idx_r,:]
    
    #------------------------------------------ 
    def _split_xy(self,X,y,y_hat,feature, threshold):
        Xy = np.column_stack((X,y,y_hat))
        Xy_l, Xy_r = self._split(Xy, feature, threshold)
        y_l     = Xy_l[:,-2]
        y_r     = Xy_r[:,-2]
        y_hat_l = Xy_l[:,-1]
        y_hat_r = Xy_r[:,-1]    
        return Xy_l[:,:-2], y_l, y_hat_l, Xy_r[:,:-2], y_r, y_hat_r
    #------------------------------------------
    def _split_y(self,X,y,y_hat,feature, threshold):
        Xy = np.column_stack((X,y,y_hat))
        Xy_l, Xy_r = self._split(Xy, feature, threshold)
        y_l     = Xy_l[:,-2]
        y_r     = Xy_r[:,-2]
        y_hat_l = Xy_l[:,-1]
        y_hat_r = Xy_r[:,-1]    
        return y_l, y_hat_l, y_r, y_hat_r
    
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
    
#---------------------------------------------------
class TestClassificationXGBTRee(BasicXGBTree):
    #------------------------------------------ 
    def _best_threshold(self,X,y, y_hat, feature):

        thresholds = np.unique(X[:,feature])
        gains      = np.zeros(thresholds.size)

        for i,threshold in enumerate(thresholds):                    
            y1,y_hat1, y2, y_hat2 = self._split_y(X,y,y_hat,feature,threshold)                    

            if (y1.size>0) and (y2.size>0):
                gains[i] = self._info_gain(y,y_hat, 
                                           y1, y_hat1, 
                                           y2, y_hat2)
            else:
                 gains[i] = 0

        idx_max = np.argmax(gains)            
        return thresholds[idx_max], gains[idx_max]

    #------------------------------------------ 
    def _best_split(self,X, y, y_hat):
        samples, features = X.shape
        gains      = np.zeros(features)
        thresholds = np.zeros(features) 

        for feature in range(features):
            thresholds[feature], gains[feature] = self._best_threshold(X,y,y_hat, feature)

        best_feature = np.argmax(gains)  # best index    

        return best_feature,thresholds[best_feature], gains[best_feature]    
    
    #------------------------------------------ 
    def _info_gain(self,y,y_hat, y_l, y_hat_l, y_r, y_hat_r):
        l = self._loss(y,y_hat)
        l_l = self._loss(y_l,y_hat_l)
        l_r = self._loss(y_r,y_hat_r)
        return 0.5*(l_r+l_l - l) + self.gamma
    
    #------------------------------------------ 
    def _loss(self,y,y_hat):
        G = np.sum(self.loss_func.gradient(y, y_hat))
        H = np.sum(self.loss_func.hessian(y, y_hat))
        return G**2/(H+self.lambda_)
    
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
    
#-------------------------------------------------
class ClassificationXGBTRee(TestClassificationXGBTRee):
    
    def __init__(self, 
                 min_samples_split = 2,
                 min_samples_leaf  = 1,
                 min_gain          = 1e-7,
                 max_depth         = -1,
                 lambda_           = 0.0, 
                 gamma             = 0.0, 
                 class_threshold   = 0.5,
                 loss_func         = None):
        
        super(ClassificationXGBTRee,self).__init__(min_samples_split,
                                                   min_samples_leaf,
                                                   min_gain,
                                                   max_depth,
                                                   lambda_, 
                                                   gamma, 
                                                   loss_func)
        self.class_threshold = class_threshold
    
    #------------------------------------------
    def _leaf_weight(self, y, y_hat):
        G = np.sum(self.loss_func.gradient(y, y_hat))
        H = np.sum(self.loss_func.hessian(y, y_hat))
        return -G/(H+self.lambda_)

    #------------------------------------------
    def get_weights(self, x, y_previous = None):
        y_f = np.asarray([self._predict(sample) for sample in x])
        if y_previous is None:
            return y_f
        else:
            return y_f + y_previous
        
    #------------------------------------------
    def predict(self, x, y_previous = None): 
        return self.loss_func.activation(self.get_weights(x, y_previous))>= self.class_threshold
    
    #---------------------------------
    def score(self, X, y, y_previous = None):
        yhat  = self.predict(X, y_previous)
        return sum((yhat==y)*1)/y.size