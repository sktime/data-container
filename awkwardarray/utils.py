from awkward import IndexedArray, JaggedArray
from scipy.ndimage.interpolation import shift
import pandas as pd
import numpy as np


def awkward_build(data_frame: pd.DataFrame) -> JaggedArray:
    """Builds an Awkward Array from a nested DataFrame.
    """
    num_cases = len(data_frame)
    cases = [None] * num_cases

    for case_num in np.arange(num_cases):
        case = data_frame.iloc[case_num, :]
        case_len = len(case)
        features = [None] * case_len

        for feature_num in np.arange(case_len):
            numpy_array = case.iloc[feature_num]
            array_size = numpy_array.shape[0]
            indexed_array = IndexedArray(np.arange(array_size), numpy_array)
            features[feature_num] = indexed_array

        cases[case_num] = JaggedArray.fromiter(features)

    return JaggedArray.fromiter(cases)


def awkward_tabularise(jagged_array: JaggedArray) -> np.ndarray:
    """Tabularises a 2-D JaggedArray of sub-arrays.

    Parameters
    ----------
    jagged_array : JaggedArray
        The The 2-D array whose sub-arrays should be tabularised.

    Returns
    -------
    np.ndarray
        jagged_array tabularised.
    """
    return np.vstack([case.flatten() for case in jagged_array])


def awkward_generate_index(jagged_array: JaggedArray) -> JaggedArray:
    """Gets an array of integers giving the index values for the sub-arrays of a specified Awkward Array.

    Parameters
    ----------
    jagged_array : JaggedArray
        The 2-D array whose sub-arrays that index values should be returned for.

    Returns
    -------
        A 2-D array who sub-arrays contain integers giving the index values for the sub-arrays of jagged_array.
    """
    index_array = jagged_array.ones_like()
    index_array.content.content = np.cumsum(index_array.content.content) - 1

    mod_array = np.cumsum(shift(index_array.count().content, 1, cval=0))
    mod_array[0] = mod_array[1]

    index_array.content = index_array.content % mod_array
    return index_array


def awkward_slope_func(jagged_array: JaggedArray) -> JaggedArray:
    """Calculates the linear slope for a specified Jagged Array.

    Parameters
    ----------
    jagged_array : JaggedArray
        The 2-D array whose sub-arrays that slope values should be returned for.

    Returns
    -------
    JaggedArray
        A 2-D array of slope values for jagged_array.
    """
    x = awkward_generate_index(jagged_array)
    y = jagged_array

    n = x.count()
    sum_x = x.sum()
    sum_x_x = (x ** 2).sum()
    sum_x_y = (x * y).sum()
    sum_y = y.sum()

    return (n * sum_x_y - sum_x * sum_y) / ((n * sum_x_x) - (sum_x * sum_x))
