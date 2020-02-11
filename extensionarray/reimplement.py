import numpy as np
import pandas as pd

from sktime.transformers.segment import IntervalSegmenter, check_is_fitted
from sktime.transformers.base import BaseTransformer
from sktime.utils.validation.supervised import validate_X

from sklearn.utils import check_random_state

class RandomIntervalSegmenter(IntervalSegmenter):
    def __init__(self, n_intervals='sqrt', min_length=2, random_state=None):
        if not isinstance(min_length, int):
            raise ValueError(f"Min_lenght must be an integer, but found: "
                             f"{type(min_length)}")
        if min_length < 1:
            raise ValueError(f"Min_lenght must be an positive integer (>= 1), "
                             f"but found: {min_length}")

        self.min_length = min_length
        self.n_intervals = n_intervals
        self.random_state = random_state

        super(RandomIntervalSegmenter, self).__init__()

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

        slices = [X.slice_time(np.arange(start=a, stop=b)).to_frame() for (a, b) in self.intervals_]

        for s, i in zip(slices, self.intervals_):
            # TODO: make sure there are no duplicate names
            s.rename(columns={colname: f"{colname}_{i[0]}_{i[1]}"}, inplace=True)

        return pd.concat(slices, axis=1)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state
        self._rng = check_random_state(random_state)

    def _rand_intervals_rand_n(self, x):
        starts = []
        ends = []
        m = x.shape[0]  # series length
        W = self._rng.randint(1, m, size=int(np.sqrt(m)))
        for w in W:
            size = m - w + 1
            start = self._rng.randint(size, size=int(np.sqrt(size)))
            starts.extend(start)
            for s in start:
                end = s + w
                ends.append(end)
        return np.column_stack([starts, ends])

    def _rand_intervals_fixed_n(self, x, n_intervals):
        len_series = len(x)
        # compute number of random intervals relative to series length (m)
        # TODO use smarter dispatch at construction to avoid evaluating if-statements here each time function is called
        if np.issubdtype(type(n_intervals), np.integer) and (n_intervals >= 1):
            pass
        elif n_intervals == 'sqrt':
            n_intervals = int(np.sqrt(len_series))
        elif n_intervals == 'log':
            n_intervals = int(np.log(len_series))
        elif np.issubdtype(type(n_intervals), np.floating) and (n_intervals > 0) and (n_intervals <= 1):
            n_intervals = int(len_series * n_intervals)
        else:
            raise ValueError(f'Number of intervals must be either "random", "sqrt", a positive integer, or a float '
                             f'value between 0 and 1, but found {n_intervals}.')

        # make sure there is at least one interval
        n_intervals = np.maximum(1, n_intervals)

        starts = self._rng.randint(len_series - self.min_length + 1, size=n_intervals)
        if n_intervals == 1:
            starts = [starts]  # make it an iterable

        ends = [start + self._rng.randint(self.min_length, len_series - start + 1) for start in starts]
        return np.column_stack([starts, ends])


class RowwiseTransformer(BaseTransformer):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        validate_X(X)

        # fitting - this transformer needs no fitting
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        return self._apply_rowwise(self.func, X, y)

    def inverse_transform(self, X, y=None):
        if not hasattr(self.transformer, 'inverse_transform'):
            raise NotImplementedError("Inverse transform not yet implemented")

    def _apply_rowwise(self, func, X, y=None):
        check_is_fitted(self, '_is_fitted')
        Xt = pd.concat([col.apply(func) for _, col in X.items()], axis=1)
        # Xt.__class__ = TimeFrame   #
        return Xt
