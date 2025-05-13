import numpy as np

def integrate_positions_from_velocity(real_data_array, filtered_data, dt=0.01):
    if not isinstance(real_data_array, np.ndarray):
        raise TypeError("Measurement must be a numpy array")
    
    integrated_data = np.cumsum(
        filtered_data[:, :, 1::2] * dt, axis=0
    )

    offset = np.mean(real_data_array, axis=0) - np.mean(integrated_data, axis=0)
    integrated_data += offset

    new_data = filtered_data.copy()
    new_data[:, :, 0::2] = integrated_data
    
    return new_data
