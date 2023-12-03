from datetime import timedelta, datetime
from typing import *

import numpy as np
import pprint
from terminaltables import AsciiTable

from .config import Config, config_to_dict
from .metrics import MetricStats
from .utils import NOT_SET
from .typing_ import *


__all__ = [
    'format_key_values', 'format_duration', 'format_as_asctime',
    'MetricsFormatter',
]


def format_key_values(key_values: Union[Dict,
                                        Config,
                                        Iterable[Tuple[str, Any]]],
                      title: Optional[str] = None,
                      formatter: Callable[[Any], str] = pprint.pformat,
                      delimiter_char: str = '=',
                      sort_keys: bool = False
                      ) -> str:
    """
    Format key value sequence into str.

    The basic usage, to format a :class:`Config`, a dict or a list of tuples:

    >>> print(format_key_values(Config(a=123, b=Config(value=456))))
    a         123
    b.value   456
    >>> print(format_key_values({'a': 123, 'b': {'value': 456}}))
    a   123
    b   {'value': 456}
    >>> print(format_key_values([('a', 123), ('b', {'value': 456})]))
    a   123
    b   {'value': 456}

    To add a title and a delimiter:

    >>> print(format_key_values(Config(a=123, b=Config(value=456)),
    ...                         title='short title'))
    short title
    =============
    a         123
    b.value   456
    >>> print(format_key_values({'a': 123, 'b': {'value': 456}},
    ...                         title='long long long title'))
    long long long title
    ====================
    a   123
    b   {'value': 456}

    Args:
        key_values: The sequence of key values, may be a :class:`Config`,
            a dict, or a list of (key, value) pairs.
            If it is a :class:`Config`, it will be flatten via
            :meth:`Config.to_flatten_dict()`.
        title: If specified, will prepend a title and a horizontal delimiter
            to the front of returned string.
        formatter: The function to format values.
        delimiter_char: The character to use for the delimiter between title
            and config key values.
        sort_keys: Whether to sort keys?

    Returns:
        The formatted str.
    """
    if len(delimiter_char) != 1:
        raise ValueError(f'`delimiter_char` must be one character: '
                         f'got {delimiter_char!r}')

    if isinstance(key_values, Config):
        key_values = config_to_dict(key_values, flatten=True)

    if hasattr(key_values, 'items'):
        data = [(key, formatter(value)) for key, value in key_values.items()]
    else:
        data = [(key, formatter(value)) for key, value in key_values]

    if sort_keys:
        data.sort(key=lambda v: v[0])

    # use the terminaltables.AsciiTable to format our key values
    table = AsciiTable(data)
    table.padding_left = 0
    table.padding_right = 3
    table.inner_column_border = False
    table.inner_footing_row_border = False
    table.inner_heading_row_border = False
    table.inner_row_border = False
    table.outer_border = False
    lines = [line.rstrip() for line in table.table.split('\n')]

    # prepend a title
    if title is not None:
        max_length = max(max(map(len, lines)), len(title))
        delim = delimiter_char * max_length
        lines = [title, delim] + lines

    return '\n'.join(lines)


def format_duration(duration: Union[float, int, timedelta],
                    precision: int = 0,
                    count_down: bool = False) -> str:
    """
    Format given time duration as human readable text.

    >>> format_duration(0)
    '0s'
    >>> format_duration(0, count_down=True)
    '0s'
    >>> format_duration(-1)
    '1s ago'
    >>> format_duration(-1, count_down=True)
    '1s ago'
    >>> format_duration(0.01, precision=2)
    '0.01s'
    >>> format_duration(0.01, precision=2, count_down=True)
    '0.01s'
    >>> format_duration(1.00, precision=2)
    '1s'
    >>> format_duration(1.00, precision=2, count_down=True)
    '1s'
    >>> format_duration(1.125)
    '1s'
    >>> format_duration(1.125, count_down=True)
    '1s'
    >>> format_duration(1.1251, precision=2)
    '1.13s'
    >>> format_duration(1.1251, precision=2, count_down=True)
    '1.13s'
    >>> format_duration(1.51)
    '2s'
    >>> format_duration(1.51, count_down=True)
    '2s'
    >>> format_duration(10)
    '10s'
    >>> format_duration(59.99, precision=2)
    '59.99s'
    >>> format_duration(59.99, precision=2, count_down=True)
    '59.99s'
    >>> format_duration(59.99)
    '1m'
    >>> format_duration(59.99, count_down=True)
    '1:00'
    >>> format_duration(60)
    '1m'
    >>> format_duration(60, count_down=True)
    '1:00'
    >>> format_duration(61)
    '1m 1s'
    >>> format_duration(61, count_down=True)
    '1:01'
    >>> format_duration(3600)
    '1h'
    >>> format_duration(3600, count_down=True)
    '1:00:00'
    >>> format_duration(86400)
    '1d'
    >>> format_duration(86400, count_down=True)
    '1d 00:00:00'
    >>> format_duration(86400 + 7200 + 180 + 4)
    '1d 2h 3m 4s'
    >>> format_duration(86400 + 7200 + 180 + 4, count_down=True)
    '1d 02:03:04'
    >>> format_duration(timedelta(days=1, hours=2, minutes=3, seconds=4))
    '1d 2h 3m 4s'
    >>> format_duration(timedelta(days=1, hours=2, minutes=3, seconds=4),
    ...                 count_down=True)
    '1d 02:03:04'

    Args:
        duration: The number of seconds, or a :class:`timedelta` object.
        precision: Precision of the seconds (i.e., number of digits to print).
        count_down: Whether or not to use the "count-down" format?  (i.e.,
            time will be formatted as "__:__:__" instead of "__h __m __s".)

    Returns:
        The formatted text.
    """
    if isinstance(duration, timedelta):
        duration = duration.total_seconds()
    else:
        duration = duration
    is_ago = duration < 0
    duration = round(abs(duration), precision)

    if count_down:
        # format the time str as "__:__:__.__"
        def format_time(seconds, has_days_part):
            # first of all, extract the hours and minutes part
            residual = []
            for unit in (3600, 60):
                residual.append(int(seconds // unit))
                seconds = seconds - residual[-1] * unit

            # format the hours and minutes
            segments = []
            for r in residual:
                if not segments and not has_days_part:
                    if r != 0:
                        segments.append(str(r))
                else:
                    segments.append(f'{r:02d}')

            # break seconds into int and real number part
            seconds_int = int(seconds)
            seconds_real = seconds - seconds_int

            # format the seconds
            if segments:
                seconds_int = f'{seconds_int:02d}'
            else:
                seconds_int = str(seconds_int)
            seconds_real = f'{seconds_real:.{precision}f}'.strip('0')
            if seconds_real == '.':
                seconds_real = ''
            seconds_suffix = 's' if not segments else ''
            segments.append(f'{seconds_int}{seconds_real}{seconds_suffix}')

            # now compose the final time str
            return ':'.join(segments)
    else:
        # format the time as "__h __m __s"
        def format_time(seconds, has_days_part):
            ret = []
            for u, s in [(3600, 'h'), (60, 'm')]:
                if seconds >= u:
                    v = int(seconds // u)
                    seconds -= v * u
                    ret.append(f'{v}{s}')
            if seconds > 1e-8:
                # seconds_int = int(seconds)
                seconds_str = f'{seconds:.{precision}f}'
                if '.' in seconds_str:
                    seconds_str = seconds_str.rstrip('0').rstrip('.')
                ret.append(f'{seconds_str}s')

            if not has_days_part and not ret:
                ret.append('0s')

            return ' '.join(ret)

    if duration < 86400:
        # less then one day, just format the time
        ret = format_time(duration, has_days_part=False)
    else:
        # equal or more than one day, format the days and the time
        days = int(duration // 86400)
        duration = duration - days * 86400
        time_str = format_time(duration, has_days_part=True)
        if time_str:
            time_str = ' ' + time_str
        ret = f'{days}d{time_str}'

    if is_ago:
        ret = f'{ret} ago'

    return ret


def format_as_asctime(dt: datetime,
                      datetime_format: str = '%Y-%m-%d %H:%M:%S',
                      msec_format: str = '%03d',
                      datetime_msec_sep: str = ',') -> str:
    """
    Format datetime `dt` using the `asctime` format of the logging module.

    >>> format_as_asctime(datetime.utcfromtimestamp(1576755571.662434))
    '2019-12-19 11:39:31,662'
    >>> format_as_asctime(datetime.utcfromtimestamp(1576755571.662434),
    ...                   datetime_format='%Y-%m-%d_%H-%M-%S',
    ...                   datetime_msec_sep='_')
    '2019-12-19_11-39-31_662'

    Args:
        dt: The datetime object.
        datetime_format: The format str for the datetime part (i.e.,
            year, month, day, hour, minute, second).
        msec_format: The format str for the milliseconds part.
        datetime_msec_sep: The separator between the datetime str and the
            milliseconds str.

    Returns:
        The formatted datetime and milliseconds.
    """
    msec = int(round(dt.microsecond / 1000))
    datetime_str = dt.strftime(datetime_format)
    msec_str = msec_format % msec
    return f'{datetime_str}{datetime_msec_sep}{msec_str}'


class MetricsFormatter(object):
    """
    Class to sort and format metric statistics into string.

    >>> fmt = MetricsFormatter()
    >>> fmt.format({
    ...     'val_loss': 1.25,
    ...     'train_loss': {'mean': 1.333333333333, 'std': 0.6666666666667},
    ...     'train_time': MetricStats(mean=2, var=2, std=1.4142135623730951),
    ... })
    'train_loss: 1.33333 (±0.666667); val_loss: 1.25; train_time: 2s (±1.414s)'
    """

    SEPARATORS: Tuple[str, str] = (': ', '; ')
    """
    Default separators, where the first is the separator between the 
    name of a metric and its value, and the second is the separator between
    different metrics.
    """

    def _metric_sort_key(self, name):
        parts = name.split('_')
        prefix_order = {'train': 0, 'val': 1, 'valid': 2, 'test': 3,
                        'pred': 4, 'predict': 5, 'epoch': 6, 'batch': 7}
        suffix_order = {'time': 9998, 'timer': 9999}
        return (suffix_order.get(parts[-1], 0), prefix_order.get(parts[0], 0),
                name)

    def _format_value(self, name: str, val: Any) -> str:
        if np.shape(val) == ():
            name_suffix = name.lower().rsplit('_', 1)[-1]
            if name_suffix in ('time', 'timer'):
                return format_duration(val, precision=3)
            else:
                return f'{float(val):.6g}'
        else:
            return str(val)

    def sorted_names(self, names: Iterable[str]) -> List[str]:
        """
        Sort the metric names.

        Args:
            names: The metric names.

        Returns:
            The sorted metric names.
        """
        return sorted(names, key=self._metric_sort_key)

    def format_metric(self, name: str, val: Any, sep: str = NOT_SET) -> str:
        """
        Format a named metric.

        >>> fmt = MetricsFormatter()
        >>> fmt.format_metric('loss', 1.25, ': ')
        'loss: 1.25'
        >>> fmt.format_metric('acc', {'mean': 0.875, 'std': 0.125}, ' = ')
        'acc = 0.875 (±0.125)'
        >>> fmt.format_metric('epoch_time', MetricStats(mean=2.5, std=1, var=1))
        'epoch_time: 2.5s (±1s)'
        >>> fmt.format_metric('value', {'mean': np.array([1, 2]), 'std': None})
        'value: [1 2]'

        Args:
            name: Name of the metric.
            val: Value of the metric, may be a number, a dict of
                ``{'mean': ..., 'std': ...}``, or an instance of
                :class:`MetricStats`.
            sep: The separator between the name and the value.
                If not specified, use ``self.DELIMIETERS[0]``.

        Returns:
            The formatted metric.
        """
        if sep is NOT_SET:
            sep = self.SEPARATORS[0]

        # if `val` is a dict with "mean" and "std"
        if isinstance(val, dict) and 'mean' in val and \
                (len(val) == 1 or (len(val) == 2 and 'std' in val)):
            mean, std = val['mean'], val.get('std')
        # elif `val` is a MetricStats object
        elif isinstance(val, MetricStats):
            mean, std = val.mean, val.std
        # else we treat `val` as a simple value
        else:
            mean, std = val, None

        # format the value part
        if std is None:
            val_str = self._format_value(name, mean)
        else:
            val_str = f'{self._format_value(name, mean)} ' \
                      f'(±{self._format_value(name, std)})'

        # now construct the final str
        return f'{name}{sep}{val_str}'

    def format(self,
               metrics: Mapping[str, Union[MetricValue,
                                           Mapping[str, MetricValue],
                                           MetricStats]],
               known_names: Optional[Sequence[str]] = None,
               sep: Tuple[str, str] = NOT_SET) -> str:
        """
        Format the given metrics.

        >>> fmt = MetricsFormatter()
        >>> fmt.format({
        ...     'acc': 0.75, 'loss': {'mean': 0.875, 'std': 0.125},
        ...     'train_time': 1.5
        ... }, known_names=['loss'], sep=(' = ', ' | '))
        'loss = 0.875 (±0.125) | acc = 0.75 | train_time = 1.5s'

        Args:
            metrics: A dict of metric values or statistics.
            known_names: Known metric names.  These metrics will be placed
                in front of other unknown metrics.
            sep: The first str is the separator between a name of metric
                and its value, while the second str is the separator
                between different metrics.  Defaults to ``self.SEPARATORS``.

        Returns:
            The formatted metrics.
        """
        if sep is NOT_SET:
            sep = self.SEPARATORS

        buf = []
        name_val_sep, metrics_sep = sep
        fmt = lambda name: self.format_metric(name, metrics[name], name_val_sep)

        # format the metrics with known names (thus preserving the known orders)
        for name in (known_names or ()):
            if name in metrics:
                buf.append(fmt(name))

        # format the metrics with unknown names (sorted by `sorted_names`)
        known_names = set(known_names or ())
        for name in self.sorted_names(metrics):
            if name not in known_names:
                buf.append(fmt(name))

        return metrics_sep.join(buf)
