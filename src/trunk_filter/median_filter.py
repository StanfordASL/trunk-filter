import numpy as np
from collections import deque

from trunk_filter.filter import Filter
    
class CausalMedianFilter(Filter):
    def __init__(self, window_size, num_nodes=3,dim_z=3):
        self.dim_z = dim_z
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.windows = [[deque(maxlen=window_size) for _ in range(dim_z)] for _ in range(num_nodes)]
    
    def update(self, z):
        assert type(z) == np.ndarray, "Measurement must be a numpy array"
        assert z.shape == (self.num_nodes, self.dim_z), "Measurement must be of shape (3,)"
        
        for i, windows in enumerate(self.windows):
            for j, window in enumerate(windows):
                window.append(z[i,j])

        new_values = np.empty((self.num_nodes, self.dim_z))
        for i, windows in enumerate(self.windows):
            for j, window in enumerate(windows):
                new_values[i,j] = np.median(window)
        
        return new_values