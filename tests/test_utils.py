import os
import numpy as np
import pandas as pd

from trunk_filter.utils import get_data_array_from_dataframe, get_dataframe_from_data_array, integrate_positions_from_velocity


script_folder = os.path.dirname(os.path.abspath(__file__))
real_data_path = os.path.join(script_folder, 'data.csv')
real_data = pd.read_csv(real_data_path)

def test_invariance():
    arr = get_data_array_from_dataframe(real_data.copy())
    inv = get_dataframe_from_data_array(arr, join_with=real_data.copy())
    assert inv.equals(real_data), "The reconstructed DataFrame is not identical to the original DataFrame"

if __name__ == "__main__":
    test_invariance()
    print("All tests passed!")