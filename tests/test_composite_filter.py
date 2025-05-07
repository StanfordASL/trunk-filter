from trunk_filter import TrunkFilter
import numpy as np
import pandas as pd
import os
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

def test_real_data():
    filter = TrunkFilter()

    # Simulate some real data
    script_folder = os.path.dirname(os.path.abspath(__file__))
    real_data_path = os.path.join(script_folder, 'states.csv')
    real_data = pd.read_csv(real_data_path)

    real_data_array = real_data[['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3']].to_numpy().reshape(-1, 3, 3)

    filtered_data = filter.update_from_array(real_data_array)

    print(filtered_data.shape)

    plt.figure(figsize=(10, 6))
    i,j = 0,0
    plt.plot(filtered_data[:, i, j], label=f'Node {i+1} Dimension {j+1}')
    plt.plot(real_data_array[:, i, j], label=f'Original Node {i+1} Dimension {j+1}', linestyle='--')
    plt.legend()

    plt.savefig('filtered_data_x.png')

    plt.figure(figsize=(10, 6))
    i,j = 2,1
    plt.plot(filtered_data[:, i, j], label=f'Node {i+1} Dimension {j+1}')
    plt.legend()
    plt.savefig('filtered_data_vx.png')

if __name__ == "__main__":
    test_update()
    test_real_data()
    print("All tests passed!")