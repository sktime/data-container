from awkwardarray.transformers import RandomIntervalSegmenter, FeatureUnionTransformer, UniversalFunctionTransformer, GenericFunctionTransformer
from awkwardarray.utils import awkward_build, awkward_slope_func
from sklearn.tree import DecisionTreeClassifier
from sktime.datasets import load_gunpoint
from sktime.pipeline import Pipeline
import pandas as pd


if __name__ == "__main__":
    # Load the gunpoint dataset and then create different size subsets

    X_small = load_gunpoint(return_X_y=False)
    X_medium = pd.concat([X_small for _ in range(20)])
    X_large = pd.concat([X_small for _ in range(200)])

    # Choose a dataset to use for the analysis and separate the class values

    X = X_medium.copy()
    y = X['class_val']
    X.drop('class_val', axis=1, inplace=True)

    # Convert the nested DataFrame into the JaggedArray to use when testing

    X = awkward_build(X)

    # Apply a random interval segmenter

    segmenter = RandomIntervalSegmenter("sqrt")
    X_segmented = segmenter.fit_transform(X)

    # Calculate the mean for the array

    mean_transformer = UniversalFunctionTransformer("mean")
    X_mean = mean_transformer.fit_transform(X_segmented)

    # Calculate the std for the array

    std_transformer = UniversalFunctionTransformer("std")
    X_std = std_transformer.fit_transform(X_segmented)

    # Calculate the slope for the array

    slope_transformer = GenericFunctionTransformer(awkward_slope_func, True)
    X_slope = slope_transformer.fit_transform(X_segmented)

    # Combine the above features into a single Numpy array

    transformers = [('mean', UniversalFunctionTransformer("mean")), ('std', UniversalFunctionTransformer("std")), ('slope', GenericFunctionTransformer(awkward_slope_func, True))]
    feature_transformer = FeatureUnionTransformer(transformers)
    features_combined = feature_transformer.fit_transform(X_segmented)

    # Fit a classifier to the combined data

    clf = DecisionTreeClassifier()
    clf.fit(features_combined, y)

    # Perform all of the above operations into a single pipeline

    steps = [
        ('segment', RandomIntervalSegmenter('sqrt')),
        ('transform', FeatureUnionTransformer([('mean', UniversalFunctionTransformer("mean")), ('std', UniversalFunctionTransformer("std")), ('slope', GenericFunctionTransformer(awkward_slope_func, True))])),
        ('clf', DecisionTreeClassifier())
    ]

    base_estimator = Pipeline(steps, random_state=1)
    base_estimator.fit(X, y)
