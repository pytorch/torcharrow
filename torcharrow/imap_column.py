# Copyright (c) Facebook, Inc. and its affiliates.
import abc
from dataclasses import dataclass

import torcharrow.dtypes as dt

from .icolumn import IColumn

# -----------------------------------------------------------------------------
# IMapColumn


class IMapColumn(IColumn):
    def __init__(self, device, dtype):
        assert dt.is_map(dtype)
        super().__init__(device, dtype)
        # must be set by subclasses
        self.maps: IMapMethods = None


# -----------------------------------------------------------------------------
# MapMethods


class IMapMethods(abc.ABC):
    """Vectorized list functions for IListColumn"""

    def __init__(self, parent):
        self._parent: IMapColumn = parent

    @abc.abstractmethod
    def keys(self):
        """
        Return a list of keys for each map entry.

        See Also
        --------
        maps.values - Returns list of values for each map entry.

        Examples
        --------
        >>> import torcharrow as ta
        >>> mf = ta.Column([
        >>>  {'helsinki': [-1.3, 21.5], 'moscow': [-4.0,24.3]},
        >>>  {'algiers':[11.2, 25.2], 'kinshasa':[22.2,26.8]}
        >>>  ])
        >>> mf.maps.keys()
        0  ['helsinki', 'moscow']
        1  ['algiers', 'kinshasa']
        dtype: List(string), length: 2, null_count: 0
        """
        pass

    @abc.abstractmethod
    def values(self):
        """
        Return a list of values for each map entry.

        See Also
        --------
        maps.keys - Returns list of keys for each map entry.

        Examples
        --------
        >>> import torcharrow as ta
        >>> mf = ta.Column([
        >>>  {'helsinki': [-1.3, 21.5], 'moscow': [-4.0,24.3]},
        >>>  {'algiers':[11.2, 25.2], 'kinshasa':[22.2,26.8]}
        >>>  ])
        >>> mf.maps.values()
        0  [[-1.3, 21.5], [-4.0, 24.3]]
        1  [[11.2, 25.2], [22.2, 26.8]]
        dtype: List(List(float64)), length: 2, null_count: 0
        """
        pass

    def get(self, i, fill_value):
        me = self._parent

        def fun(xs):
            # TODO improve perf by looking at lists instead of first building a map
            return xs.get(i, fill_value)

        return me._vectorize(fun, me.dtype.item_dtype)


# ops on maps --------------------------------------------------------------
#  'get',
#  'items',
#  'keys',
#  'pop',
#  'popitem',
#  'setdefault',
#  'update',
#  'values'
