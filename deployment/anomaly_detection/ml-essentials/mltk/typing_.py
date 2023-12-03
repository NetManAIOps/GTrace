import re
from typing import *

import numpy as np
from bson import ObjectId

__all__ = [
    'TConfig', 'TObject', 'TValue',
    'MetricValue', 'PatternType', 'EllipsisType', 'CommandOrArgsType',
    'DocumentType', 'FilterType', 'ExperimentId',
    'Array', 'ArrayTuple', 'ArrayTupleOrList', 'ArraysOrArray',
    'ArraysOrArrayGenerator', 'XYArrayTuple', 'ArrayDType',
    'ArrayShape', 'BatchGenerator',
]

TConfig = TypeVar('TConfig')
"""Generic type of config classes."""

TObject = TypeVar('TObject')
"""Generic type of arbitrary object classes."""

TValue = TypeVar('TValue')
"""Value type of a mapping."""

PatternType = type(re.compile('x'))
"""The type of regex patterns."""

EllipsisType = type(...)
"""The type of ellipsis "..."."""

MetricValue = Union[float, np.ndarray]
"""The metric value type, i.e., a float number oer a float array."""

CommandOrArgsType = Union[List[str], Tuple[str, ...], str]
"""Command line or argument list type."""

DocumentType = Dict[str, Any]
"""Experiment document type."""

FilterType = Dict[str, Any]
"""Experiment filter type."""

ExperimentId = Union[ObjectId, str]
"""Experimnent ID type."""

Array = Union[np.ndarray, Any]
"""The type of array-like objects."""

ArrayTuple = Tuple[Array, ...]
"""Tuple of arrays."""

ArrayTupleOrList = Union[Tuple[Array, ...], List[Array]]
"""Tuple or list of arrays."""

ArraysOrArray = Union[ArrayTupleOrList, Array]
"""Tuple or list of arrays, or an array."""

ArraysOrArrayGenerator = Generator[ArraysOrArray, None, None]
"""Generator that produces arrays or sequences of arrays."""

XYArrayTuple = Tuple[np.ndarray, np.ndarray]
"""Tuple of two arrays, the x and y arrays."""

ArrayDType = Union[np.dtype, Type]
"""NumPy dtype or Python type for arrays."""

ArrayShape = Tuple[int, ...]
"""Array shape type."""

BatchGenerator = Generator[
    Union[int, Tuple[int, ArrayTuple]],
    None,
    None
]
"""The return type of `iter_batches()`."""
