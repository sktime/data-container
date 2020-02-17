from awkwardarray.utils import awkward_slope_func, awkward_tabularise, awkward_arrays_differ, awkward_generate_dummy_data
from sktime.utils.time_series import time_series_slope
from sklearn.tree import DecisionTreeClassifier
from awkward import JaggedArray
import numpy as np


if __name__ == "__main__":
    # Get a very small dummy dataset for illustrative purposes

    X, y = awkward_generate_dummy_data()

    # Access specific cases or features

    jagged_array_cases = X[:2, :]
    jagged_array_features = X[:, :1]

    # Tabularise the data (only works if all series are of the same length, but this is the current assumption of the existing method)

    jagged_array_tabularised = awkward_tabularise(X)

    # Apply some ufuncs to all of the time series

    jagged_array_sum_all = X.sum()
    jagged_array_max_all = X.max()
    jagged_array_exp_all = np.exp(X)

    jagged_array_mean_all = X.mean()
    jagged_array_std_all = X.std()

    # Apply some ufuncs to specific rows or columns of time series

    jagged_array_sum_some = X[:2].sum()
    jagged_array_exp_some = np.exp(X[:2])

    # Apply the slope function to each time series iteratively (this is no equivalent of pandas.Series.apply() in Awkward Array)

    jagged_array_slope_func = JaggedArray.fromiter([[time_series_slope(time_series) for time_series in case] for case in X])

    # Apply the slope function to each time series using ufuncs

    jagged_array_slope = awkward_slope_func(X)

    # Compare the results from applying the sktime and our own slope function

    jagged_array_slopes_differ = awkward_arrays_differ(jagged_array_slope, jagged_array_slope_func, 1e-10)

    # Combine multiple single value features into one (Numpy) array

    to_combine = [jagged_array_mean_all, jagged_array_std_all, jagged_array_slope]
    combined = np.hstack([jagged_array.regular() for jagged_array in to_combine])

    # Apply a Scikit-learn classifier

    clf = DecisionTreeClassifier()
    clf.fit(combined, y=y)

    test_data = combined[:1]
    clf_predict = clf.predict(test_data)
