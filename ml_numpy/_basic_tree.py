import numpy as np
import matplotlib.pyplot as plt 

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
        
        
class BasicTree:
    def __init__(self, 
                 min_samples_split = 2,
                 min_samples_leaf  = 1,
                 min_gain          = 1e-7,
                 max_depth         = -1):
        
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
    
    