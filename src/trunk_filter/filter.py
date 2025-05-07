import numpy as np

class Filter:
    def __init__(self):
        raise NotImplementedError
    
    def update(self, z):
        raise NotImplementedError

    def update_from_array(self, z_array):
        values = []
        for z in z_array:
            values.append(self.update(z))
        return np.array(values)
