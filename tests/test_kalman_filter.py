from trunk_filter.kalman_filter import KalmanFilter, RTSFilter
import numpy as np

measurements = np.array([
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    ] + [
        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ]
    ] * 1000)

# Test the Kalman filter implementation
# Ensure that the Kalman filter converges to the expected measured state which is held constant
def test_kalman_filter():
    filter = KalmanFilter(num_nodes=3, dt=0.01, measurement_noise=1e-6, position_process_noise=1e-4, velocity_process_noise=2e-2)

    measurement = measurements[0]

    for _ in range(1000):
        new_state = filter.update(measurement)

    assert np.allclose(new_state[:,::2], measurement), "The new state does not match the expected measurement."


def test_rts_filter():
    filter = RTSFilter(
        KalmanFilter(num_nodes=3, dt=0.01, measurement_noise=1e-6, position_process_noise=1e-4, velocity_process_noise=2e-2)
    )

    new_states = filter.update_from_array(measurements)


def test_update_from_array():
    filter = KalmanFilter(num_nodes=3, dt=0.01, measurement_noise=1e-6, position_process_noise=1e-4, velocity_process_noise=2e-2)
    
    new_states = filter.update_from_array(measurements)
    assert np.allclose(new_states[-1,:,::2], measurements[-1]), "The new states do not match the expected measurements."

if __name__ == "__main__":
    test_kalman_filter()
    test_rts_filter()
    test_update_from_array()
    print("All tests passed!")