import warnings

import numpy as np
import pandas as pd
from pandas import Series

from extensionarray.array import TimeDtype, TimeArray

_SERIES_WARNING_MSG = """\
    You are passing non-time series data to the TimeSeries constructor. Currently,
    it falls back to returning a pandas Series. But in the future, we will start
    to raise a TypeError instead."""


def is_time_series_type(data):
    """
    Check if the data is of time series dtype.

    Does not include simple numpy array.
    """
    if isinstance(getattr(data, "dtype", None), TimeDtype):
        # TimeArray, TimeSeries and Series[TimeArray]
        return True
    else:
        return False


class TimeSeries(Series):
    """
    A Series object designed to store shapely geometry objects.

    Parameters
    ----------
    data : array-like, dict, scalar value
        The geometries to store in the GeoSeries.
    index : array-like or Index
        The row index for the TimeSeries.
    time_index : array-like or Index (optional)
        The index denoting the relative position of each observation in each time series
    kwargs
        Additional arguments passed to the Series constructor,
         e.g. ``name``.

    See Also
    --------
    TimeSeriesFrame
    pandas.Series

    """

    _metadata = ["name"]

    def __init__(self, data=None, index=None, time_index=None, **kwargs):
        name = kwargs.pop("name", None)

        if isinstance(data, np.ndarray):
            data = TimeArray(data, time_index)

        if not is_time_series_type(data):
            # if data is None and dtype is specified (eg from empty overlay
            # test), specifying dtype raises an error:
            # https://github.com/pandas-dev/pandas/issues/26469
            kwargs.pop("dtype", None)
            # Use Series constructor to handle input data
            s = pd.Series(data, index=index, name=name, **kwargs)
            # prevent trying to convert non-time series objects
            if s.dtype != object:
                if s.empty:
                    s = s.astype(object)
                else:
                    warnings.warn(_SERIES_WARNING_MSG, FutureWarning, stacklevel=2)
                    return s
            index = s.index
            name = s.name

        super(TimeSeries, self).__init__(data, index=index, name=name, **kwargs)

    def tabularise(self, return_array=False):
        return self._values.tabularise(self.name, return_array)

    def tabularize(self, return_array=False):
        return self.tabularise(return_array)

    def slice_time(self, time_index, inplace=False):
        if inplace:
            # TODO: enable inplace slicing
            raise NotImplementedError("inplace slicing of time series is not supported yet")

        return TimeSeries(self._values.slice_time(time_index), index=self.index, name=self.name)

    @property
    def has_common_index(self):
        return self._values.has_common_index

    @property
    def time_index(self):
        return self._values.time_index

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        if len(self) == 0:
            return self._constructor(dtype=self.dtype, index=self.index).__finalize__(
                self
            )

        # dispatch to agg
        if isinstance(func, (list, dict)):
            raise NotImplementedError("Simultaneous application of multiple functions" ""
                                      "not supported yet")

        # if we are a string, try to dispatch
        if isinstance(func, str):
            raise NotImplementedError("Dispatch by function name not supported yet")

        # handle ufuncs and lambdas
        if kwds or args and not isinstance(func, np.ufunc):
            def f(x):
                return func(x, *args, **kwds)
        else:
            f = func

        with np.errstate(all="ignore"):
            if isinstance(f, np.ufunc):
                raise NotImplementedError("np.ufunc not supported yet")

            # row-wise access
            # TODO: define when to use aggregate and when to use map
            mapped = self._values.aggregate(f)

        # TODO: define what type to return
        return pd.Series(data=mapped, index=self.index, name=self.name)

    @property
    def _constructor(self):
        return TimeSeries

    @property
    def _constructor_expanddim(self):
        from extensionarray.timeframe import TimeFrame

        return TimeFrame