import numpy as np
import matplotlib.pyplot as plt 

from ._basic_tree import BasicTree

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

#--------------------------------
class XGBClassifier:
    def __init__(self,
                 max_estimators    = 1,
                 min_samples_split = 2,
                 min_samples_leaf  = 1,
                 min_gain          = 1e-7,
                 max_depth         = -1,
                 lambda_           = 0.0, 
                 gamma             = 0.0, 
                 class_threshold   = 0.5,
                 min_score         = 1e-5,
                 loss_func         = None):
        
        self.max_estimators    = max_estimators
        self.min_samples_split = min_samples_split       
        self.min_samples_leaf  = min_samples_leaf     
        self.min_gain          = min_gain      
        self.max_depth         = max_depth      
        self.lambda_           = lambda_
        self.gamma             = gamma
        self.class_threshold   = class_threshold
        
        self.loss_func = loss_func
        
        if self.loss_func is None:
            self.loss_func = LogLossWithLogits()
        
        self.min_score = min_score
        self.trees     = []
        self.n_trees   = 0
    
    #----------------------------------    
    def fit(self,X,y, X_val=None,y_val=None, verbose=True):
        y_hat = np.zeros_like(y)
        self.trees = np.array([])

        if (X_val is None and y_val is None):
            val_flag  = False
        else :
            val_flag  = True
            y_hat_val = np.zeros_like(y_val)

        
        for i in range(self.max_estimators):
            tree = ClassificationXGBTRee(min_samples_split = self.min_samples_split,
                                         min_samples_leaf  = self.min_samples_leaf,
                                         min_gain          = self.min_gain,
                                         max_depth         = self.max_depth,
                                         lambda_           = self.lambda_,
                                         gamma             = self.gamma,
                                         class_threshold   = self.class_threshold,
                                         loss_func         = self.loss_func
                                         ).fit(X,y,y_hat)
            

            y_hat = tree.get_weights(X,y_hat)
            self.trees = np.append(self.trees, tree)

            if verbose:
                print('='*10)
                if val_flag:
                    y_hat_val = tree.get_weights(X_val,y_hat_val)
                    print('i = ',i,'; train score = %.4f'%tree.score(X,y,y_hat),  'val score = %.4f'%tree.score(X_val,y_val,y_hat_val))
                else:
                    print('i = ',i,'; score = %.4f'%tree.score(X,y,y_hat))

        self.n_trees = i+1
        return self
    #--------------------
    def predict(self,X):

        y_hat = np.zeros(X.shape[0])

        for i in range(self.n_trees):
            y_hat = self.trees[i].get_weights(X,y_hat)

        return self.loss_func.activation(y_hat)>= self.class_threshold
    
    #--------------------
    def score(self, X, y):
        yhat  = self.predict(X)
        return sum((yhat==y)*1)/y.size
    
    #--------------------
    def print_scores(self, X, y):
        y_hat = np.zeros(X.shape[0])

        for i in range(self.n_trees):
            y_hat = self.trees[i].get_weights(X,y_hat)   
            print('i = ',i,'; train score = %.4f'%self.trees[i].score(X,y,y_hat))
        
        print('overall score = %.4f'%self.score(X,y))
    
    
#--------------------------------
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
    
    
#--------------------------------
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
    #------------------------------------------
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
#-------------------------------------
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
    
#----------------------------------------
