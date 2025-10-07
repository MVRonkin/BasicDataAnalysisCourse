import numpy as np
import matplotlib.pyplot as plt 

from ._tree_id3 import DecisionTreeID3

class DecisionTreeС45(DecisionTreeID3):
    def _info_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        G = self._entropy(y) - p * self._entropy(y1) - (1 - p) * self._entropy(y2)
        return G/-(p * np.log2(p) + (1-p) * np.log2(1-p)) 
