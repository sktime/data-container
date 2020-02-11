import numbers
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from collections.abc import Iterable



# -----------------------------------------------------------------------------
# Extension Type
# -----------------------------------------------------------------------------

class TimeBase(object):
    """
    the type of CategoricalDtype, this metaclass determines subclass ability
    """
    pass


class TimeDtype(ExtensionDtype):
    name = "timeseries"
    type = TimeBase
    kind = 'O'
    na_value = np.nan

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from '{}'".format(cls, string))

    @classmethod
    def construct_array_type(cls):
        return TimeArray


# ----------------------------------------------------------------------------------------------------------------------
# Extension Container
# ----------------------------------------------------------------------------------------------------------------------

class TimeArray(ExtensionArray):
    """
    Class wrapping a numpy array of time series and
    holding the array-based implementations.
    """

    # A note on the internal data layout. TimeArray implements a
    # collection of equal length time series in which rows correspond to
    # a single observed time series. We use a Numpy array to store the
    # individual data points of each time series, i.e. columns relate to
    # specific points in time. The time indices are stored in a separate
    # Numpy array, either of dimension 1xT for a common index across
    # time series or NxT for a separate index for each time series.
    _dtype = TimeDtype()
    _can_hold_na = True

    def __init__(self, data, time_index=None):
        """
        Initialise a new TimeArray containing equal length time series

        Parameters
        ----------
        data : numpy.array, TimeArray
            Equal length (or padded) time series object in which each row corresponds to an time series
        time_index : numpy.array, pandas.Series, list of integers, optional (default=None)
            A common time index for all time series (1xT array, where T is the number of time steps) or separate
            time indices with one time index per time series (NxT array, where N is the number of time series). If no
            time index is provided, a common default integer index [0, 1, ..., T-1] is created for all time series.
        """
        if isinstance(data, self.__class__):
            data = data.data
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "'data' should be a numpy.array"
            )

        if time_index is None:
            n_timesteps = data.shape[1] if len(data.shape) == 2 else 1
            time_index = pd.Int64Index(np.arange(start=0, stop=n_timesteps))
        elif isinstance(time_index, list):
            time_index = pd.Int64Index(time_index)
        elif isinstance(time_index, np.ndarray):
            if not time_index.shape[0] == 1 or time_index.shape[0] == data.shape[1]:
                raise TypeError(
                    "If 'time_index' is supplied as a np.array, it should either be a 1xD or a NxD array."
                )
            elif not issubclass(time_index.dtype.type, np.integer):
                raise TypeError(
                    "'time_index' must be an array of integers, instead got {}".format(time_index.dtype)
                )

            time_index = pd.Int64Index(time_index)
        elif not isinstance(time_index, pd.Index):
            raise TypeError(
                "'time_index' should either be a list of integers, an array of integers, a pandas.Series"
                "or None."
            )

        if not time_index.shape == (data.shape[1],):
            raise TypeError(
                "'time_index' must either be missing or must be as long as the time series."
            )

        self.data = data
        self.time_index = time_index

        self.check_equal_index()
        self._contains_missing = np.any(self.isna())

    @classmethod
    def _from_ndarray(cls, data, copy=False):
        """Zero-copy construction of a TimeArray from an ndarray.

        Parameters
        ----------
        data : ndarray
            The raw data of the time series as ndarray
        copy : bool, default False
            Whether to copy the data

        Returns
        -------
        ExtensionArray
        """
        if copy:
            data = data.copy()
        new = TimeArray([])
        new.data = data
        return new

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        raise NotImplementedError("Construction of TimeArray from a scalar sequence has not been implemented "
                                  "yet.")

    @classmethod
    def _from_factorized(cls, values, original):
        raise NotImplementedError("Reconstruction of TimeArray after factorization has not been implemented "
                                  "yet.")

    @classmethod
    def _concat_same_type(cls, to_concat):
        for ta in to_concat:
            if not ta.time_index.equals(to_concat[0].time_index):
                raise ValueError("Time indices of concatenated TimeArrays must all be equal.")

        data = np.vstack([ta.data for ta in to_concat])
        return TimeArray(data)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def has_common_index(self):
        return self._equal_index

    # -------------------------------------------------------------------------
    # Interfaces
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self.data[idx]
        elif isinstance(idx, (Iterable, slice)):
            return TimeArray(self.data[idx])
        else:
            raise TypeError("Index type not supported", idx)

    def __setitem__(self, key, value):
        if isinstance(value, pd.Series):
            value = value.values

        if isinstance(value, TimeArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("cannot set a single element with an array")
            self.data[key] = value.data
        elif isinstance(value, np.ndarray):
            if isinstance(key, (list, np.ndarray)):
                value_array = np.empty(1, dtype=object)
                value_array[:] = [value]
                self.data[key] = value_array
            else:
                self.data[key] = value
        else:
            raise TypeError(
                "Value should be either a TimeSeriesBase or None, got %s" % str(value)
            )

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return 1

    # -------------------------------------------------------------------------
    # time series functionality
    # -------------------------------------------------------------------------

    def tabularise(self, name=None, return_array=False):
        if name is None:
            name = "dim"

        if return_array:
            return self.data

        return pd.DataFrame(self.data, columns=[name + "_" + str(i) for i in self.time_index])

    def tabularize(self, return_array=False):
        return self.tabularise(return_array)

    def check_equal_index(self):
        if len(self.time_index.shape) == 1:
            self._equal_index = True
        else:
            arr = self.time_index.to_numpy()
            self._equal_index = (arr == arr[0]).all()

        return self._equal_index

    def slice_time(self, time_index):
        if len(self.time_index.shape) == 1:
            sel = np.isin(self.time_index.to_numpy(), time_index)
            return TimeArray(self.data[:, sel], time_index=self.time_index[sel])
        elif self._equal_index:
            sel = np.isin(self.time_index.to_numpy()[0, :], time_index)
            return TimeArray(self.data[:, sel], time_index=self.time_index[:, sel])
        else:
            raise NotImplementedError("Time slicing that results in unequal lengths has not been implemented yet.")

    # ------------------------------------------------------------------------
    # Ops
    # ------------------------------------------------------------------------

    def isna(self):
        """Indicator for whether an entire time series is missing.

        Only fully missing time series, that is time series with every observation missing, are considered missing.

        Returns
        -------
        Boolean array of shape = [n_instances]

        Examples
        --------
        >>> TimeArray(numpy.array([[1., 1.], [1., np.nan], [np.nan, np.nan]])).isna()
        array([False, False, True])

        """
        return np.apply_over_axes(np.all, self.na_grid(), 1)

    def na_grid(self):
        """Return an indicator array marking all time points at which values are missing.

        Returns
        -------
        Boolean array of shape = [n_instances, n_timesteps]
        """
        return np.isnan(self.data)

    def isin(self, other):
        """Check whether elements of `self` are in `other`.

        Parameters
        ----------
        other :

        Returns
        -------
        contained : ndarray
            A 1-D boolean ndarray with the same length as self.
        """
        raise NotImplementedError("isin() not implemented for TimeArray")

    def _fill(self, idx, value):
        """ Fill index locations with value

        Value should be a scalar of the same type or compatible with the internally held data
        """

        # Add input check

        self.data[idx] = np.array([value], dtype=object)
        return self

    def fillna(self, value=None, method=None, limit=None):
        """ Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in the time series. Contrary to the
            behaviour in scalar types, the method is applied within a single
            cell (i.e. a single time series).
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        Returns
        -------
        filled : ExtensionArray with NA/NaN filled
        """

        if method is not None:
            raise NotImplementedError("fillna with a method is not yet supported")

        mask = self.na_grid()
        new_values = self.copy()

        if mask.any():
            # fill with value
            new_values = new_values._fill(mask, value)

        return new_values

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """

        raise NotImplementedError("astype has not been implemented yet")

    # -------------------------------------------------------------------------
    # general array like compat
    # -------------------------------------------------------------------------

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
        """
        n_row = self.data.shape[0]
        n_time = self.data.shape[1]
        irreg = np.zeros(n_time + 1)

        return np.array([irreg] + [np.array(self.data[i]) for i in range(n_row)])[1:]
        #return self.data

    def copy(self, *args, **kwargs):
        return TimeArray(self.data.copy())

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = 0

        result = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        if fill_value == 0:
            result[result == 0] = None
        return TimeArray(result)

    @property
    def nbytes(self):
        return self.data.nbytes


    # ------------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        from pandas.io.formats.printing import format_object_summary

        template = "{class_name}{data}\nLength: {length}, dtype: {dtype}({equal} index)"
        data = format_object_summary(
            self, self._formatter(), indent_for_name=False
        ).rstrip(", \n")
        class_name = "<{}>\n".format(self.__class__.__name__)

        if self._equal_index:
            equal_text = "equal"
        else:
            equal_text = "unequal, max"

        return template.format(
            class_name=class_name, data=data, length=len(self), dtype=self.dtype, equal=equal_text,
        )

    def _formatter(self, boxed: bool = False) -> Callable[[Any], Optional[str]]:
        """Formatting function for scalar values.
        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.
        Parameters
        ----------
        boxed : bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).
        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """
        if boxed:
            return str
        return repr

    # ------------------------------------------------------------------------
    # Test section of potential includes
    # ------------------------------------------------------------------------

    def __add__(self, o):
        if np.all(self.time_index != o.time_index):
            raise ValueError("The time indices of two TimeArrays that should be added must be identical.")

        return TimeArray(self.data + o.data, time_index=self.time_index)

    def map(self, mapper):
        """Map time series using input correspondence (dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.

        Returns
        -------
        TimeArray
            Mapped time series.

        """
        mapped = np.apply_along_axis(mapper, axis=1, arr=self.data)
        time_index = None

        if mapped.shape == self.data.shape:
            time_index = self.time_index

        if mapped.ndim == 1:
            mapped = np.expand_dims(mapped, axis=1)

        # TODO: deal with mappings similar to pandas, distinguishing agg, transform, apply
        return type(self)(data=mapped, time_index=time_index)

    def aggregate(self, func):

        return np.apply_along_axis(func, axis=1, arr=self.data)