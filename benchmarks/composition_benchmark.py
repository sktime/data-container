import timeit

SETUP = """
import pandas as pd
import numpy as np
from sktime.datasets import load_gunpoint
from sklearn.preprocessing import FunctionTransformer
from sktime.utils.time_series import time_series_slope
from sktime.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
{packages}

# Load the original dataset and create different sizes
X_base = load_gunpoint(return_X_y=False)
X_small = {X}
X_medium = pd.concat([X_small for _ in range(20)])
X_large = pd.concat([X_small for _ in range(200)])

# Choose the size used for this experiment
X = X_{size}.copy()

# Extract y into a separate vector
y = X['class_val']
X.drop('class_val', axis=1, inplace=True)
"""

PACKAGES = """
from sktime.transformers.segment import RandomIntervalSegmenter
from sktime.transformers.compose import RowwiseTransformer
"""

SMALL = SETUP.format(packages=PACKAGES, X="X_base", size="small")

repeats = 5
runs = 0

# **********************************************************************************************************************
# Time the transformations individually

SEGM = """
segmenter = RandomIntervalSegmenter(n_intervals=3)
X_segm = segmenter.fit_transform(X)
"""
timeit.repeat(setup=SMALL, stmt=SEGM, repeat=repeats, number=runs)


MEAN = """
mean_transformer = RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))
X_mean = mean_transformer.fit_transform(X_segm)
"""
timeit.repeat(setup=SMALL + SEGM, stmt=MEAN, repeat=repeats, number=runs)



STD = """
std_transformer = RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))
X_std = std_transformer.fit_transform(X_segm)
"""
timeit.repeat(setup=SMALL + SEGM + MEAN, stmt=STD, repeat=repeats, number=runs)


SLOPE = """
slope_transformer = RowwiseTransformer(FunctionTransformer(func=time_series_slope, validate=False))
X_slope = slope_transformer.fit_transform(X_segm)
"""
timeit.repeat(setup=SMALL + SEGM + MEAN + STD, stmt=SLOPE, repeat=repeats, number=runs)


UNION = """
X_union = pd.concat([X_mean, X_std, X_slope], axis=1)
"""
timeit.repeat(setup=SMALL + SEGM + MEAN + STD + SLOPE, stmt=UNION, repeat=repeats, number=runs)


TREE = """
dt = DecisionTreeClassifier()
dt.fit(X_union, y)
"""
timeit.repeat(setup=SMALL + SEGM + MEAN + STD + SLOPE + UNION, stmt=TREE, repeat=repeats, number=runs)



# **********************************************************************************************************************
# Time the entire pipeline at once

PIPELINE_SETUP = """
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
"""

PIPELINE_RUN = """
base_estimator.fit(X, y)
"""

timeit.repeat(setup=SMALL + PIPELINE_SETUP, stmt=PIPELINE_RUN, repeat=repeats, number=runs)
