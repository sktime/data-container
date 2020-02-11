import numpy as np
import pandas as pd
from collections import abc
from pandas import DataFrame
from extensionarray.array import TimeDtype
from extensionarray.timeseries import TimeSeries

class TimeFrame(DataFrame):
    """
    A TimeFrame object is a pandas.DataFrame that has one or more columns
    containing time series.
    """

    _metadata = []

    def __init__(self, *args, **kwargs):

        if "data" in kwargs:
            data = kwargs['data']
        elif len(args) > 0 and isinstance(args[0], abc.Iterable) and not isinstance(args[0], (str, bytes)):
            data = args[0]


        # TODO: find a more elegant solution
        if "data" in kwargs or len(args) > 0 and isinstance(args[0], abc.Iterable) and not isinstance(args[0], (str, bytes, TimeSeries)):
            if isinstance(data, dict):
                for i in data.keys():
                    if isinstance(data[i], np.ndarray):
                        data[i] = TimeSeries(data[i])
            elif isinstance(data, abc.Iterable) and not isinstance(data, (str, bytes, np.ndarray)):
                for i in range(len(data)):
                    if isinstance(data[i], np.ndarray):
                        data[i] = TimeSeries(data[i])

        super(TimeFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return TimeFrame

    @property
    def _constructor_sliced(self):
        return TimeSeries

    def __getitem__(self, key):
        """
        If the result is a column containing time series, return a
        TimeSeries. If it's a DataFrame with a time series column, return a
        TimeFrame.
        """
        result = super(TimeFrame, self).__getitem__(key)
        ts_idxs = [isinstance(d, TimeDtype) for d in self.dtypes]
        ts_cols = self.columns[ts_idxs]
        if isinstance(key, str) and key in ts_cols:
            result.__class__ = TimeSeries
        elif isinstance(result, DataFrame) and np.isin(ts_cols, result.columns):
            result.__class__ = TimeFrame
        elif isinstance(result, DataFrame) and not np.isin(ts_cols, result.columns):
            result.__class__ = DataFrame
        return result

    def tabularise(self):

        return pd.concat([i.tabularise() if isinstance(i, TimeSeries) else i for _, i in self.items()], axis=1)