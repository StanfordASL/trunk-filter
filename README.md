# Trunk Filter

Trunk Filter is a package that allows users of the ASL Trunk to filter readings from the Mocap system. Specifically, a causal median filter and kalman filter is provided.

## Features

- **Kalman Filter**: Implements a Kalman filter for smoothing and predicting multidimensional data.
- **Median Filter**: Implements a causal median filter for robust filtering of noisy data.
- **Composite Filter**: Combines multiple filters (e.g., Kalman and median filters) for enhanced performance.
- **Trunk Filter**: We provide a tuned composite filter tailored to the trunk robot @ ASL
## Installation

```
pip install trunk-filter
```

## Usage

```python
from trunk_filter import TrunkFilter

f = TrunkFilter()

# provide measurement, which is a np.ndarray of shape (num_nodes, num_states_per_node), e.g. (3,3) for the ASL Trunk
filtered_measurement = f.update(measurement)

```
