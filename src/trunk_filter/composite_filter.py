from trunk_filter.filter import Filter
from trunk_filter.kalman_filter import KalmanFilter, RTSFilter
from trunk_filter.median_filter import CausalMedianFilter

class CompositeFilter(Filter):
    def __init__(self, *filters):
        self.filters = filters

    def update(self, z):
        for filter in self.filters:
            z = filter.update(z)

        return z
    
    def update_from_array(self, z_array):
        for filter in self.filters:
            z_array = filter.update_from_array(z_array)
            
        return z_array
    
class TrunkFilter(CompositeFilter):
    def __init__(self, window_size=3, measurement_noise=1e-2, position_process_noise=1e-4, velocity_process_noise=1000e-2):
        super().__init__(
            KalmanFilter(num_nodes=3, measurement_noise=measurement_noise, position_process_noise=position_process_noise, velocity_process_noise=velocity_process_noise),
            #CausalMedianFilter(num_nodes=3, dim_z=6, window_size=window_size),
        )

class NonCausalTrunkFilter(CompositeFilter):
    def __init__(self, window_size=3, measurement_noise=1e-2, position_process_noise=1e-4, velocity_process_noise=1000e-2):
        super().__init__(
            RTSFilter(
                KalmanFilter(num_nodes=3, measurement_noise=measurement_noise, position_process_noise=position_process_noise, velocity_process_noise=velocity_process_noise)
            )
        )