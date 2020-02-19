from awkwardarray.utils import awkward_tabularize, awkward_detabularize
from sktime.transformers.base import BaseTransformer
from awkward import JaggedArray
from sklearn.exceptions import NotFittedError
from scipy.ndimage.interpolation import shift
from typing import List, Tuple
import numpy as np


class RandomIntervalSegmenter(BaseTransformer):
    """Transformer that segments time-series into random intervals (which may overlap and there may be duplicates) with random starting points and lengths.
    """
    def __init__(self, n_intervals: str or int, min_length: int = 2):
        """
        Parameters
        ----------
        method : str
            The segmentation method to be used (only sqrt currently supported).

        min_length : int
            The minimum length of an interval.
        """
        self._n_intervals = n_intervals
        self._min_length = min_length
        self._is_fitted = False

    @property
    def n_intervals(self) -> str or int:
        """
        Returns
        -------
        str or int
            The segmentation method to be used (only sqrt currently supported) to calculate the number of intervals to use, or the number of intervals to use.
        """
        return self._n_intervals

    @property
    def min_length(self) -> int:
        """
        Returns
        -------
        int
            The minimum length of an interval.
        """
        return self._min_length

    def fit(self, x: JaggedArray, y=None):
        """Empty fit function that does nothing.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        y : None
            None as it is transformer on X.

        Returns
        -------
        self : object
            Returns self.
        """
        self._is_fitted = True
        return self

    def transform(self, x, y=None):
        """
        Transform X, segments time-series in each column into random intervals using interval indices generated
        during `fit`.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        Returns
        -------
        JaggedArray
            The result of segmenting x.
        """

        # Get the length of the time series; as in the existing implementation, this assumes that they are all the same length

        rng = np.random.mtrand._rand
        len_series = len(x[0, 0])

        # Calculate the number of intervals

        if "sqrt" == self._n_intervals:
            num_intervals = np.maximum(1, int(np.sqrt(len_series)))

        elif isinstance(self._n_intervals, int):
            num_intervals = self._n_intervals

        else:
            raise ValueError(f"Unsupported _n_intervals '{self._n_intervals}'")

        # Generate the starts and ends for the intervals

        starts = rng.randint(len_series - self._min_length + 1, size=num_intervals)

        if num_intervals == 1:
            starts = [starts]

        starts = np.array(starts)

        ends = [start + rng.randint(self._min_length, len_series - start + 1) for start in starts]
        intervals = zip(starts, ends)

        # Get the data for the features

        tabularized = awkward_tabularize(x)
        num_cases = len(tabularized)
        interval_lengths = ends - starts
        features = []

        for start, end in intervals:
            feature = tabularized[:, start:end]
            feature_array = feature.content
            feature_array = feature_array.reshape((num_cases, -1))
            features.append(feature_array)

        # Generate the content array for the array to return

        content_array_content = np.hstack(features)
        content_array_content = content_array_content.reshape(-1)
        offsets = shift(np.resize(interval_lengths, num_intervals * num_cases + 1).cumsum(), 1)
        content_array = JaggedArray.fromoffsets(offsets, content_array_content)

        # Generate the array to return

        num_intervals_array = np.array([num_intervals])
        offsets = shift(np.resize(num_intervals_array, num_cases + 1).cumsum(), 1)
        to_return = JaggedArray.fromoffsets(offsets, content_array)

        return to_return


class TabularTransformer(BaseTransformer):
    """A transformer that turns time series/panel data into tabular data.

    This is useful for transforming time-series/panel data into a format that is accepted by standard validation learning algorithms (as in sklearn).
    """
    def __init__(self):
        self._counts = None
        self._content_counts = None

    def fit(self, x: JaggedArray, y=None):
        """Empty fit function that does nothing.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        y : None
            None as it is transformer on X.

        Returns
        -------
        self : object
            Returns self.
        """
        self._is_fitted = True
        return self

    def transform(self, x: JaggedArray, y=None) -> np.ndarray:
        """Tabularizes a specified JaggedArray.

        Parameters
        ----------
        x : JaggedArray
            The array to tabularize.

        y : None
            None as it is transformer on x.

        Returns
        -------
        JaggedArray
            x tabularised.
        """
        self._counts = x.counts
        self._content_counts = x.content.counts

        return awkward_tabularize(x)

    def inverse_transform(self, x: JaggedArray, y=None):
        """Detabularizes a specified Numpy Array.

        Parameters
        ----------
        x : JaggedArray
            The array to detabularize.

        y : None
            None as it is transformer on x.

        Returns
        -------
        JaggedArray
            x detabularised.
        """
        if self._counts is None or self._content_counts is None:
            raise NotFittedError("This has not yet been fitted in transform")
        else:
            return awkward_detabularize(x, self._content_counts, self._counts)


class UniversalFunctionTransformer(BaseTransformer):
    """A convenience wrapper that applies a Universal Function (ufunc) defined as an instance method to an Awkward array.
    """
    def __init__(self, u_func: str or np.ufunc):
        """
        Parameters
        ----------
        u_func : str or np.ufunc
            The name of the ufunc to apply if the function is an instance method, or the function defined in Numpy to apply.
        """
        if not isinstance(u_func, str) and not isinstance(u_func, np.ufunc):
            raise ValueError("u_func is not a str or a numpy.ufunc")
        else:
            self._numpy_method = True if isinstance(u_func, np.ufunc) else False
            self._u_func = u_func
            self._is_fitted = False

    @property
    def u_func(self):
        """
        Returns
        -------
        str or np.ufunc
            The name of the ufunc to apply if the function is an instance method, or the function defined in Numpy to apply.
        """
        return self._u_func

    def fit(self, x: JaggedArray, y=None):
        """Empty fit function that does nothing.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        y : None
            None as it is transformer on x.

        Returns
        -------
        self : object
            Returns self.
        """
        self._is_fitted = True
        return self

    def transform(self, x: JaggedArray, y=None) -> JaggedArray:
        """Apply the Universal Function to x.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        y : None
            None as it is transformer on x.

        Returns
        -------
        JaggedArray
            The result of applying the Universal Function to x.
        """
        if self._numpy_method:
            return self._u_func(x)
        else:
            u_func = getattr(x, self._u_func)
            return u_func()

    def inverse_transform(self, X, y=None):
        """Not implemented for this type of transformer.
        """
        raise NotImplementedError("Transformer does not have an inverse transform method")


class GenericFunctionTransformer(BaseTransformer):
    """A convenience wrapper that applies a provided function to an Awkward array.
    """
    def __init__(self, func: callable, apply_to_container: bool):
        """
        Parameters
        ----------
        func : callable
            The function to apply.

        apply_to_container : bool
            True if the function should be applied to the array provided during training, False if it should be applied to the sub-arrays of this array.
        """
        if callable(func):
            self._func = func
            self._apply_to_container = apply_to_container
            self._is_fitted = False
        else:
            raise ValueError("func is not a callable")

    @property
    def func(self):
        """
        Returns
        -------
        callable
            The function to apply.
        """
        return self._func

    @property
    def apply_to_container(self):
        """
        Returns
        -------
        bool
            True if the function should be applied to the array provided during training, False if it should be applied to the sub-arrays of this array.
        """
        return self._apply_to_container

    def fit(self, x: JaggedArray, y=None):
        """Empty fit function that does nothing.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        y : None
            None as it is transformer on x.

        Returns
        -------
        self : object
            Returns self.
        """
        self._is_fitted = True
        return self

    def transform(self, x: JaggedArray, y=None) -> JaggedArray:
        """Apply the Universal Function to x.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        y : None
            None as it is transformer on x.

        Returns
        -------
        JaggedArray
            The result of applying the function to x.
        """
        if self._apply_to_container:
            return self._func(x)
        else:
            return JaggedArray.fromiter([[self._func(time_series) for time_series in case] for case in x])

    def inverse_transform(self, X, y=None):
        """Not implemented for this type of transformer.
        """
        raise NotImplementedError("Transformer does not have an inverse transform method")


class FeatureUnionTransformer(BaseTransformer):
    """Combines the outputs of one or more transformers that output an Awkward Array (all with the same number of cases) and concatenates them in a single Numpy array.
    """
    def __init__(self, transformers: List[Tuple[str, BaseTransformer]]):
        """
        Parameters
        ----------
        transformers : List[Tuple[str, object]]
            The sequence of transformers (and their corresponding label) to apply.
        """
        self._transformers = transformers
        self._is_fitted = False

    @property
    def transformers(self):
        """
        Returns
        -------
        List[Tuple[str, object]]
            The sequence of transformers (and their corresponding label) to apply.
        """
        return self._transformers

    def fit(self, x: JaggedArray, y=None):
        """Empty fit function that does nothing.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        y : None
            None as it is transformer on x.

        Returns
        -------
        self : object
            Returns self.
        """
        for _, transformer in self._transformers:
            transformer.fit(x, y)

        self._is_fitted = True
        return self

    def transform(self, x: JaggedArray, y=None) -> np.ndarray:
        """Apply the Universal Function to x.

        Parameters
        ----------
        x : JaggedArray
            The training input samples.

        y : None
            None as it is transformer on x.

        Returns
        -------
        np.ndarray
            The result of applying the function to x.
        """
        return np.hstack([transformer.transform(x).regular() for _, transformer in self._transformers])

    def inverse_transform(self, X, y=None):
        """Apply the `fit_transform()` method of the transformer on each row.
        """
        raise NotImplementedError("Transformer does not have an inverse transform method")
