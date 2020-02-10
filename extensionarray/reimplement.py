import numpy as np
import pandas as pd

from sktime.transformers.segment import IntervalSegmenter, check_is_fitted
from sktime.transformers.base import BaseTransformer
from sktime.utils.data_container import concat_nested_arrays
from sktime.utils.data_container import tabularize
from sktime.utils.validation.supervised import validate_X


class RandomIntervalSegmenter(IntervalSegmenter):
    def __init__(self, n_intervals='sqrt', min_length=2, random_state=None):
        super(IntervalSegmenter, self).__init__(n_intervals='sqrt', min_length=2, random_state=None)

    def fit(self, X, y=None):

        col = X.columns[0]
        X = X[col]
        self.input_shape_ = X.shape

        if not X.has_common_index:
            raise ValueError("All time series in transform column {} must share a common time index".format(col))
        self._time_index = X.time_index

        if self.n_intervals == 'random':
            self.intervals_ = self._rand_intervals_rand_n(self._time_index)
        else:
            self.intervals_ = self._rand_intervals_fixed_n(self._time_index, n_intervals=self.n_intervals)

        return self

    def transform(self, X, y=None):
        colname = X.columns[0]
        X = X[colname]

        # Check inputs.
        check_is_fitted(self, 'intervals_')

        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns of input is different from what was seen'
                             'in `fit`')

        slices = [X.slice_time(np.arange(start=a, stop=b)) for (a, b) in self.intervals_]

        for s, i in zip(slices, self.intervals_):
            s.name = f"{colname}_{i[0]}_{i[1]}"

        return pd.concat(slices, axis=1)


class RowwiseTransformer(BaseTransformer):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        validate_X(X)

        # fitting - this transformer needs no fitting
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        func = self.transformer.fit_transform
        return self._apply_rowwise(func, X, y)

    def inverse_transform(self, X, y=None):
        if not hasattr(self.transformer, 'inverse_transform'):
            raise AttributeError("Transformer does not have an inverse transform method")

        func = self.transformer.inverse_transform
        return self._apply_rowwise(func, X, y)

    def _apply_rowwise(self, func, X, y=None):
        check_is_fitted(self, '_is_fitted')
        Xt = X.apply(self.transformer.fit_transform)
        return Xt