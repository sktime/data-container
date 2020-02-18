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


def awkward_tabularize(jagged_array: JaggedArray) -> np.ndarray:
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
    return jagged_array.flatten().regular()


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


def awkward_arrays_differ(array_a: JaggedArray, array_b: JaggedArray, threshold: float) -> bool:
    """Determines if two specified Awkward arrays differ.

    Parameters
    ----------
    array_a : JaggedArray
        The 1st 2-D array whose sub-arrays should be compared.

    array_b : JaggedArray
        The 2nd 2-D array whose sub-arrays should be compared.

    threshold : float
        A float value giving the maximum value that array elements can differ before the arrays are considered different.

    Returns
    -------
    bool
        True if a pair of elements in array_a and array_b differ by more than threshold, False otherwise.
    """
    array_diff = (array_a - array_b)
    array_diff_max = np.abs(array_diff.max())

    return (array_diff_max > threshold).any()


def awkward_generate_dummy_data() -> (JaggedArray, JaggedArray):
    """Generates a small amount of dummy data that can be used to clearly see the output of functions.

    Returns
    -------
    (JaggedArray, JaggedArray)
        A tuple containing 2 JaggedArrays.
        The 1st array is a 2-D array whose sub-arrays contain IndexedArray representing time series; cases are represented by the rows and features by the columns.
        The 2nd array contains the classification values (0 or 1) for each case.
    """
    x_tab_0 = IndexedArray([0, 1, 2], [0.0, 1.0, 2.0])
    x_tab_1 = IndexedArray([0, 1, 2], [1.0, 2.0, 3.0])
    x_tab_2 = IndexedArray([0, 1, 2], [2.0, 3.0, 4.0])
    x_tab_3 = IndexedArray([0, 1, 2], [3.0, 4.0, 5.0])

    y_tab_0 = IndexedArray([0, 1, 2], [4.0, 5.0, 6.0])
    y_tab_1 = IndexedArray([0, 1, 2], [5.0, 6.0, 7.0])
    y_tab_2 = IndexedArray([0, 1, 2], [6.0, 7.0, 8.0])
    y_tab_3 = IndexedArray([0, 1, 2], [6.0, 7.0, 8.0])

    case_0 = JaggedArray.fromiter([x_tab_0, y_tab_0])
    case_1 = JaggedArray.fromiter([x_tab_1, y_tab_1])
    case_2 = JaggedArray.fromiter([x_tab_2, y_tab_2])
    case_3 = JaggedArray.fromiter([x_tab_3, y_tab_3])

    y = JaggedArray.fromiter([0, 1, 1, 0])
    jagged_array = JaggedArray.fromiter([case_0, case_1, case_2, case_3])

    return jagged_array, y
