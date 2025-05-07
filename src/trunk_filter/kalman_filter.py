from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from scipy.linalg import block_diag
import numpy as np

from trunk_filter.filter import Filter

class KalmanFilter(Filter):
    def __init__(self, num_nodes=3, dt=0.01, measurement_noise=1e-6, position_process_noise=1e-4, velocity_process_noise=2e-2):
        self.dim_x = 6
        self.dim_z = 3
        self.num_nodes = num_nodes

        self.filters = [
            FilterPyKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z) for _ in range(num_nodes)
        ]
        
        A = np.array([
                [1.,dt],
                [0,1.]
            ])

        for kf in self.filters:
            kf.F = block_diag(A, A, A)
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0]
            ])

            kf.P *= 1e5 # Initial covariance is large
            kf.R = np.eye(3) * measurement_noise
            kf.Q = np.diag([position_process_noise, velocity_process_noise] * 3)


    def update(self, z):
        assert type(z) == np.ndarray, "Measurement must be a numpy array"
        assert z.shape == (self.num_nodes, self.dim_z), "Measurement must be of shape (3, 3)"

        values = np.empty((self.num_nodes, self.dim_x))

        for i, filter in enumerate(self.filters):
            filter.predict()
            filter.update(z[i])
            values[i] = filter.x.flatten()

        return values
    
""" class RTSFilter(Filter):
    def __init__(self, kalman_filter):
        self.dim_x = kalman_filter.dim_x
        self.dim_z = kalman_filter.dim_z
        self.num_nodes = kalman_filter.num_nodes

        self.filters = kalman_filter.filters
        
    def update_from_array(self, z_array):
        assert type(z_array) == np.ndarray, "Measurement must be a numpy array"
        values = []

        for i, filter in enumerate(self.filters):
            mu, cov, _, _ = filter.batch_filter
            xs, cov = filter.rts_smoother(z_array[:, i, :])
            values.append(xs)

        return np.array(values) """