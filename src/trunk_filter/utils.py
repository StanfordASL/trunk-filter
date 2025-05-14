import numpy as np
import pandas as pd
import re

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

def get_data_array_from_dataframe(df):
    links = [int(re.findall(r'\d+', col)[0]) for col in df.columns if col.startswith('x') and not col.endswith('_new')]
    query = [f'{axis}{i}' for i in links for axis in ['x', 'vx', 'y', 'vy', 'z', 'vz']]
    return df[query].to_numpy().reshape(-1, len(links), 6)

def get_dataframe_from_data_array(data_array, join_with=None):
    data = data_array.reshape(-1, 6 * data_array.shape[1])
    columns = [f'{axis}{i+1}' for i in range(data_array.shape[1]) for axis in ['x', 'vx', 'y', 'vy', 'z', 'vz']]
    
    if join_with is None:
        return pd.DataFrame(data, columns=columns)
    else:
        join_with.update(pd.DataFrame(data, columns=columns))
        return join_with

