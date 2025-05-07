from trunk_filter.median_filter import CausalMedianFilter
import numpy as np


# Test the Kalman filter implementation
# Ensure that the Kalman filter converges to the expected measured state which is held constant

def test_update():
    filter = CausalMedianFilter(num_nodes=3, dim_z=3, window_size=3)

    measurement = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    for _ in range(1000):
        new_state = filter.update(measurement)

    assert np.allclose(new_state, measurement), "The new state does not match the expected measurement."

if __name__ == "__main__":
    test_update()
    print("All tests passed!")