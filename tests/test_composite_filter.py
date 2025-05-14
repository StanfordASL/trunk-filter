from trunk_filter import TrunkFilter
from trunk_filter.composite_filter import NonCausalTrunkFilter
from trunk_filter.utils import integrate_positions_from_velocity

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def test_update():
    filter = TrunkFilter()

    measurement = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    for _ in range(1000):
        new_state = filter.update(measurement)

    assert np.allclose(new_state[:,::2], measurement), "The new state does not match the expected measurement."

def test_real_data(filter):
    # Simulate some real data
    script_folder = os.path.dirname(os.path.abspath(__file__))
    real_data_path = os.path.join(script_folder, 'states.csv')
    real_data = pd.read_csv(real_data_path)

    real_data_array = real_data[['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3']].to_numpy().reshape(-1, 3, 3)
    # Get only first 100 samples
    real_data_array = real_data_array[:200]

    filtered_data = filter.update_from_array(real_data_array)
    integrated_data = integrate_positions_from_velocity(real_data_array, filtered_data)

    i,j = 0,0
    plt.figure()
    plt.plot(real_data_array[:, i, j], label=f'Original Node {i+1} Dimension {j+1}', linestyle='--')
    plt.plot(filtered_data[:, i, j], label=f'Node {i+1} Dimension {j+1}')
    plt.plot(integrated_data[:, i, j], label=f'Integrated Node {i+1} Dimension {j+1}')
    plt.legend()
    plt.savefig(f'figures/filtered_data_x_{filter.__class__.__name__}.png')

    i,j = 2,1
    plt.figure()
    plt.plot(filtered_data[:, i, j], label=f'Node {i+1} Dimension {j+1}')
    # Velocity is not provided by the dataset
    plt.legend()
    plt.savefig(f'figures/filtered_data_vx_{filter.__class__.__name__}.png')

if __name__ == "__main__":
    test_update()
    
    for filter in tqdm([TrunkFilter(), NonCausalTrunkFilter()]): #TODO: Not pytest friendly
        test_real_data(filter)

    print("All tests passed!")