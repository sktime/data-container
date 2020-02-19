
# Import packages and disable certain warnings that impact existing Sktime code

from awkwardarray.utils import awkward_build, awkward_slope_func, awkward_arrays_differ, data_frames_differ, awkward_array_data_frame_differ
from sklearn.preprocessing import FunctionTransformer
from sktime.transformers.segment import RandomIntervalSegmenter
from sktime.transformers.compose import RowwiseTransformer, Tabularizer
from sktime.utils.time_series import time_series_slope
from sktime.datasets import load_gunpoint
import awkwardarray.transformers
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the data to analyse

X_small = load_gunpoint(return_X_y=False)
X_medium = pd.concat([X_small for _ in range(20)])
X_large = pd.concat([X_small for _ in range(200)])

X = X_medium.copy()
y = X['class_val']
X.drop('class_val', axis=1, inplace=True)

# Specify the threshold used when comparing numeric values

threshold = 1e-12

# Get the appropriate forms of the data for testing

data_frame = X
awkward_array= awkward_build(X)
assert(not awkward_array_data_frame_differ(awkward_array, data_frame, threshold))

# Apply tabularizers to the data and compare the outputs

tabularizer = awkwardarray.transformers.TabularTransformer()
awkward_tabularized = tabularizer.fit_transform(awkward_array)
awkward_detabularized = tabularizer.inverse_transform(awkward_tabularized)
assert(not awkward_arrays_differ(awkward_detabularized, awkward_array, threshold))

tabularizer = Tabularizer(check_input=False)
data_frame_tabularized = tabularizer.fit_transform(data_frame)
data_frame_detabularized = tabularizer.inverse_transform(data_frame_tabularized)
assert(not data_frames_differ(data_frame_detabularized, data_frame, threshold))

assert(not awkward_array_data_frame_differ(awkward_detabularized, data_frame_detabularized, threshold))

# Apply a random interval segmenter to the data and compare the outputs

np.random.mtrand.seed(1)
segmenter = awkwardarray.transformers.RandomIntervalSegmenter(n_intervals=3)
awkward_segmented = segmenter.fit_transform(awkward_array)

np.random.mtrand.seed(1)
segmenter = RandomIntervalSegmenter(n_intervals=3)
data_frame_segmented = segmenter.fit_transform(X)

assert(not awkward_array_data_frame_differ(awkward_segmented, data_frame_segmented, threshold))

# Calculate the mean for the data and compare the outputs

mean_transformer = awkwardarray.transformers.UniversalFunctionTransformer("mean")
awkward_mean = mean_transformer.fit_transform(awkward_segmented)

mean_transformer = RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))
data_frame_mean = mean_transformer.fit_transform(data_frame_segmented)

assert(not awkward_array_data_frame_differ(awkward_mean, data_frame_mean, threshold, False))

# Calculate the std for the data and compare the outputs

std_transformer = awkwardarray.transformers.UniversalFunctionTransformer("std")
awkward_std = std_transformer.fit_transform(awkward_segmented)

std_transformer = RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))
data_frame_std = std_transformer.fit_transform(data_frame_segmented)

assert(not awkward_array_data_frame_differ(awkward_std, data_frame_std, threshold, False))

# Calculate the slope for the data and compare the outputs

slope_transformer = awkwardarray.transformers.GenericFunctionTransformer(awkward_slope_func, True)
awkward_slope = slope_transformer.fit_transform(awkward_segmented)

slope_transformer = RowwiseTransformer(FunctionTransformer(func=time_series_slope, validate=False))
data_frame_slope = slope_transformer.fit_transform(data_frame_segmented)

assert(not awkward_array_data_frame_differ(awkward_slope, data_frame_slope, threshold, False))

slope_transformer = awkwardarray.transformers.GenericFunctionTransformer(time_series_slope, False)
awkward_slope = slope_transformer.fit_transform(awkward_segmented)

assert(not awkward_array_data_frame_differ(awkward_slope, data_frame_slope, threshold, False))

# Combine the above features into a single Numpy array and compare the outputs

awkward_union = np.hstack([transformed.regular() for transformed in [awkward_mean, awkward_std, awkward_slope]])

data_frame_union = pd.concat([data_frame_mean, data_frame_std, data_frame_slope], axis=1)
data_frame_union = data_frame_union.to_numpy()

assert(np.abs((awkward_union - data_frame_union).max()) <= threshold)
