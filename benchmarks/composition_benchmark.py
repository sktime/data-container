import timeit

SETUP_CODE = """
import pandas as pd
import numpy as np
from sktime.datasets import load_gunpoint
from sktime.transformers.segment import RandomIntervalSegmenter
from sktime.transformers.compose import RowwiseTransformer
from sktime.utils.time_series import time_series_slope
from sktime.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# Load the original dataset and create different sizes
X_small = load_gunpoint(return_X_y=False)
X_medium = pd.concat([X_small for _ in range(20)])
X_large = pd.concat([X_small for _ in range(200)])

# Choose a dataset for the analysis
X = X_medium.copy()
y = X['class_val']
X.drop('class_val', axis=1, inplace=True)
"""

repeats = 5
runs = 0

# **********************************************************************************************************************
# Time the transformations individually

UP_TO_NOW = SETUP_CODE
TEST_CODE = """
segmenter = RandomIntervalSegmenter(n_intervals='sqrt')
X_segm = segmenter.fit_transform(X)
"""
timeit.repeat(setup=UP_TO_NOW, stmt=TEST_CODE, repeat=repeats, number=runs)

UP_TO_NOW += TEST_CODE
TEST_CODE = """
mean_transformer = RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))
X_mean = mean_transformer.fit_transform(X_segm)
"""
timeit.repeat(setup=UP_TO_NOW, stmt=TEST_CODE, repeat=repeats, number=runs)


UP_TO_NOW += TEST_CODE
TEST_CODE = """
std_transformer = RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))
X_std = std_transformer.fit_transform(X_segm)
"""
timeit.repeat(setup=UP_TO_NOW, stmt=TEST_CODE, repeat=repeats, number=runs)


UP_TO_NOW += TEST_CODE
TEST_CODE = """
slope_transformer = RowwiseTransformer(FunctionTransformer(func=time_series_slope, validate=False))
X_slope = slope_transformer.fit_transform(X_segm)
"""
timeit.repeat(setup=UP_TO_NOW, stmt=TEST_CODE, repeat=repeats, number=runs)


UP_TO_NOW += TEST_CODE
TEST_CODE = """
X_union = pd.concat([X_mean, X_std, X_slope], axis=1)
"""
timeit.repeat(setup=UP_TO_NOW, stmt=TEST_CODE, repeat=repeats, number=runs)


UP_TO_NOW += TEST_CODE
TEST_CODE = """
dt = DecisionTreeClassifier()
dt.fit(X_union, y)
"""
timeit.repeat(setup=UP_TO_NOW, stmt=TEST_CODE, repeat=repeats, number=runs)



# **********************************************************************************************************************
# Time the entire pipeline at once

TEST_CODE = """
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

timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=repeats, number=runs)
