# Import packages and disable certain warnings that impact existing Sktime code

import timeit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

# Specify how many the provided code should be executed for each timing, and the number of times this process should be repeated

runs = 3
repeats = 3

# Specify the code that should get run before any timing information is calculated using Awkward Array

AWKWARD_SETUP_CODE = """
from awkwardarray.transformers import RandomIntervalSegmenter, FeatureUnionTransformer, UniversalFunctionTransformer, GenericFunctionTransformer
from awkwardarray.utils import awkward_build, awkward_slope_func, awkward_tabularize
from sklearn.tree import DecisionTreeClassifier
from sktime.datasets import load_gunpoint
from sktime.pipeline import Pipeline
import pandas as pd
import numpy as np

X_small = load_gunpoint(return_X_y=False)
X_medium = pd.concat([X_small for _ in range(20)])
X_large = pd.concat([X_small for _ in range(200)])

X = X_medium.copy()
y = X['class_val']
X.drop('class_val', axis=1, inplace=True)

X = awkward_build(X)
"""

# Apply a random interval segmenter to the dataset using Awkward Array

AWKWARD_UP_TO_NOW = AWKWARD_SETUP_CODE
AWKWARD_TEST_CODE = """
X_tabularized = awkward_tabularize(X)
"""
awkward_tabularize_timings = timeit.repeat(setup=AWKWARD_UP_TO_NOW, stmt=AWKWARD_TEST_CODE, repeat=repeats, number=runs)
awkward_tabularize_timings = [timing/runs for timing in awkward_tabularize_timings]
print(f"\nAwkward Array Tabularize: {awkward_tabularize_timings}")

# Apply a random interval segmenter to the dataset using Awkward Array

AWKWARD_UP_TO_NOW += AWKWARD_TEST_CODE
AWKWARD_TEST_CODE = """
segmenter = RandomIntervalSegmenter(n_intervals=3)
X_segmented = segmenter.fit_transform(X)
"""
awkward_segment_timings = timeit.repeat(setup=AWKWARD_UP_TO_NOW, stmt=AWKWARD_TEST_CODE, repeat=repeats, number=runs)
awkward_segment_timings = [timing/runs for timing in awkward_segment_timings]
print(f"Awkward Array Segmenter: {awkward_segment_timings}")

# Calculate the mean for the dataset using Awkward Array

AWKWARD_UP_TO_NOW += AWKWARD_TEST_CODE
AWKWARD_TEST_CODE = """
mean_transformer = UniversalFunctionTransformer("mean")
X_mean = mean_transformer.fit_transform(X_segmented)
"""
awkward_mean_timings = timeit.repeat(setup=AWKWARD_UP_TO_NOW, stmt=AWKWARD_TEST_CODE, repeat=repeats, number=runs)
awkward_mean_timings = [timing/runs for timing in awkward_mean_timings]
print(f"Awkward Array Mean: {awkward_mean_timings}")

# Calculate the std for the dataset using Awkward Array

AWKWARD_UP_TO_NOW += AWKWARD_TEST_CODE
AWKWARD_TEST_CODE = """
std_transformer = UniversalFunctionTransformer("std")
X_std = std_transformer.fit_transform(X_segmented)
"""
awkward_std_timings = timeit.repeat(setup=AWKWARD_UP_TO_NOW, stmt=AWKWARD_TEST_CODE, repeat=repeats, number=runs)
awkward_std_timings = [timing/runs for timing in awkward_std_timings]
print(f"Awkward Array Std: {awkward_std_timings}")

# Calculate the slope for the dataset using Awkward Array

AWKWARD_UP_TO_NOW += AWKWARD_TEST_CODE
AWKWARD_TEST_CODE = """
slope_transformer = GenericFunctionTransformer(awkward_slope_func, True)
X_slope = slope_transformer.fit_transform(X_segmented)
"""
awkward_slope_timings = timeit.repeat(setup=AWKWARD_UP_TO_NOW, stmt=AWKWARD_TEST_CODE, repeat=repeats, number=runs)
awkward_slope_timings = [timing/runs for timing in awkward_slope_timings]
print(f"Awkward Array Slope: {awkward_slope_timings}")

# Combine the above features into a single Numpy array using Awkward Array

AWKWARD_UP_TO_NOW += AWKWARD_TEST_CODE
AWKWARD_TEST_CODE = """
features_combined = np.hstack([transformed.regular() for transformed in [X_mean, X_std, X_slope]])
"""
awkward_union_timings = timeit.repeat(setup=AWKWARD_UP_TO_NOW, stmt=AWKWARD_TEST_CODE, repeat=repeats, number=runs)
awkward_union_timings = [timing/runs for timing in awkward_union_timings]
print(f"Awkward Array Union: {awkward_union_timings}")

# Fit a classifier to the combined data to the dataset using Awkward Array

AWKWARD_UP_TO_NOW += AWKWARD_TEST_CODE
AWKWARD_TEST_CODE = """
clf = DecisionTreeClassifier()
clf.fit(features_combined, y)
"""
awkward_classifier_timings = timeit.repeat(setup=AWKWARD_UP_TO_NOW, stmt=AWKWARD_TEST_CODE, repeat=repeats, number=runs)
awkward_classifier_timings = [timing/runs for timing in awkward_classifier_timings]
print(f"Awkward Array Classifier: {awkward_classifier_timings}")

# Perform all of the above operations into a single pipeline using Awkward Array

AWKWARD_TEST_CODE = """
steps = [
    ('segment', RandomIntervalSegmenter('sqrt')),
    ('transform', FeatureUnionTransformer([('mean', UniversalFunctionTransformer("mean")), ('std', UniversalFunctionTransformer("std")), ('slope', GenericFunctionTransformer(awkward_slope_func, True))])),
    ('clf', DecisionTreeClassifier())
]

base_estimator = Pipeline(steps, random_state=1)
base_estimator.fit(X, y)
"""
awkward_pipeline_timings = timeit.repeat(setup=AWKWARD_SETUP_CODE, stmt=AWKWARD_TEST_CODE, repeat=repeats, number=runs)
awkward_pipeline_timings = [timing/runs for timing in awkward_pipeline_timings]
print(f"Awkward Array Pipeline: {awkward_pipeline_timings}")

# Specify the code that should get run before any timing information is calculated using Awkward Array

DATAFRAME_SETUP_CODE = """
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sktime.transformers.segment import RandomIntervalSegmenter
from sktime.transformers.compose import RowwiseTransformer
from sktime.utils.time_series import time_series_slope
from sktime.pipeline import Pipeline, FeatureUnion
from sktime.utils.data_container import tabularize
from sktime.datasets import load_gunpoint
import pandas as pd
import numpy as np

X_small = load_gunpoint(return_X_y=False)
X_medium = pd.concat([X_small for _ in range(20)])
X_large = pd.concat([X_small for _ in range(200)])

X = X_medium.copy()
y = X['class_val']
X.drop('class_val', axis=1, inplace=True)
"""

# Apply a random interval segmenter to the dataset using nested DataFrames

DATAFRAME_UP_TO_NOW = DATAFRAME_SETUP_CODE
DATAFRAME_TEST_CODE = """
X_tabularized = tabularize(X,return_array=True)
"""
dataframe_tabularize_timings = timeit.repeat(setup=DATAFRAME_UP_TO_NOW, stmt=DATAFRAME_TEST_CODE, repeat=repeats, number=runs)
dataframe_tabularize_timings = [timing/runs for timing in dataframe_tabularize_timings]
print(f"\nDataFrame Tabularize: {dataframe_tabularize_timings}")

# Apply a random interval segmenter to the dataset using nested DataFrames

DATAFRAME_UP_TO_NOW += DATAFRAME_TEST_CODE
DATAFRAME_TEST_CODE = """
segmenter = RandomIntervalSegmenter(n_intervals=3)
X_segmented = segmenter.fit_transform(X)
"""
dataframe_segment_timings = timeit.repeat(setup=DATAFRAME_UP_TO_NOW, stmt=DATAFRAME_TEST_CODE, repeat=repeats, number=runs)
dataframe_segment_timings = [timing/runs for timing in dataframe_segment_timings]
print(f"DataFrame Segmenter: {dataframe_segment_timings}")

# Calculate the mean for the dataset using nested DataFrames

DATAFRAME_UP_TO_NOW += DATAFRAME_TEST_CODE
DATAFRAME_TEST_CODE = """
mean_transformer = RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))
X_mean = mean_transformer.fit_transform(X_segmented)
"""
dataframe_mean_timings = timeit.repeat(setup=DATAFRAME_UP_TO_NOW, stmt=DATAFRAME_TEST_CODE, repeat=repeats, number=runs)
dataframe_mean_timings = [timing/runs for timing in dataframe_mean_timings]
print(f"DataFrame Mean: {dataframe_mean_timings}")

# Calculate the std for the dataset using nested DataFrames

DATAFRAME_UP_TO_NOW += DATAFRAME_TEST_CODE
DATAFRAME_TEST_CODE = """
std_transformer = RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))
X_std = std_transformer.fit_transform(X_segmented)
"""
dataframe_std_timings = timeit.repeat(setup=DATAFRAME_UP_TO_NOW, stmt=DATAFRAME_TEST_CODE, repeat=repeats, number=runs)
dataframe_std_timings = [timing/runs for timing in dataframe_std_timings]
print(f"DataFrame Std: {dataframe_std_timings}")

# Calculate the slope for the dataset using nested DataFrames

DATAFRAME_UP_TO_NOW += DATAFRAME_TEST_CODE
DATAFRAME_TEST_CODE = """
slope_transformer = RowwiseTransformer(FunctionTransformer(func=time_series_slope, validate=False))
X_slope = slope_transformer.fit_transform(X_segmented)
"""
dataframe_slope_timings = timeit.repeat(setup=DATAFRAME_UP_TO_NOW, stmt=DATAFRAME_TEST_CODE, repeat=repeats, number=runs)
dataframe_slope_timings = [timing/runs for timing in dataframe_slope_timings]
print(f"DataFrame Slope: {dataframe_slope_timings}")

# Combine the above features using nested DataFrames

DATAFRAME_UP_TO_NOW += DATAFRAME_TEST_CODE
DATAFRAME_TEST_CODE = """
X_union = pd.concat([X_mean, X_std, X_slope], axis=1)
X_union = X_union.to_numpy()
"""
dataframe_union_timings = timeit.repeat(setup=DATAFRAME_UP_TO_NOW, stmt=DATAFRAME_TEST_CODE, repeat=repeats, number=runs)
dataframe_union_timings = [timing/runs for timing in dataframe_union_timings]
print(f"DataFrame Union: {dataframe_union_timings}")

# Fit a classifier to the combined data to the dataset using nested DataFrames

DATAFRAME_UP_TO_NOW += DATAFRAME_TEST_CODE
DATAFRAME_TEST_CODE = """
clf = DecisionTreeClassifier()
clf.fit(X_union, y)
"""
dataframe_classifier_timings = timeit.repeat(setup=DATAFRAME_UP_TO_NOW, stmt=DATAFRAME_TEST_CODE, repeat=repeats, number=runs)
dataframe_classifier_timings = [timing/runs for timing in dataframe_classifier_timings]
print(f"DataFrame Classifier: {dataframe_classifier_timings}")

# Perform all of the above operations into a single pipeline nested DataFrames

DATAFRAME_UP_TO_NOW += DATAFRAME_TEST_CODE
DATAFRAME_TEST_CODE = """
steps = [
    ('segment', RandomIntervalSegmenter(n_intervals='sqrt')),
    ('transform', FeatureUnion([
        ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
        ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))),
        ('slope', RowwiseTransformer(FunctionTransformer(func=time_series_slope, validate=False)))
    ])),
    ('clf', DecisionTreeClassifier())
]
base_estimator = Pipeline(steps, random_state=1)
base_estimator.fit(X, y)
"""
dataframe_pipeline_timings = timeit.repeat(setup=DATAFRAME_UP_TO_NOW, stmt=DATAFRAME_TEST_CODE, repeat=repeats, number=runs)
dataframe_pipeline_timings = [timing/runs for timing in dataframe_pipeline_timings]
print(f"DataFrame Pipeline: {dataframe_pipeline_timings}")

# Obtain the minimum (or typical best case scenario) for each test for each implementation

awkward_tabularize = np.min(awkward_tabularize_timings)
awkward_segment = np.min(awkward_segment_timings)
awkward_mean = np.min(awkward_mean_timings)
awkward_std = np.min(awkward_std_timings)
awkward_slope = np.min(awkward_slope_timings)
awkward_union = np.min(awkward_union_timings)
awkward_classifier = np.min(awkward_classifier_timings)
awkward_pipeline = np.min(awkward_pipeline_timings)

dataframe_tabularize = np.min(dataframe_tabularize_timings)
dataframe_segment = np.min(dataframe_segment_timings)
dataframe_mean = np.min(dataframe_mean_timings)
dataframe_std = np.min(dataframe_std_timings)
dataframe_slope = np.min(dataframe_slope_timings)
dataframe_union = np.min(dataframe_union_timings)
dataframe_classifier = np.min(dataframe_classifier_timings)
dataframe_pipeline = np.min(dataframe_pipeline_timings)

tabularize_multiplier = dataframe_tabularize / awkward_tabularize
segment_multiplier = dataframe_segment / awkward_segment
mean_multiplier = dataframe_mean / awkward_mean
std_multiplier = dataframe_std / awkward_std
slope_multiplier = dataframe_slope / awkward_slope
union_multiplier = dataframe_union / awkward_union
classifier_multiplier = dataframe_classifier / awkward_classifier
pipeline_multiplier = dataframe_pipeline / awkward_pipeline

print(f"\nTabularize Multiplier: {tabularize_multiplier}")
print(f"Segment Multiplier: {segment_multiplier}")
print(f"Mean Multiplier: {mean_multiplier}")
print(f"Std Multiplier: {std_multiplier}")
print(f"Slope Multiplier: {slope_multiplier}")
print(f"Union Multiplier: {union_multiplier}")
print(f"Classifier Multiplier: {classifier_multiplier}")
print(f"Pipeline Multiplier: {pipeline_multiplier}")
