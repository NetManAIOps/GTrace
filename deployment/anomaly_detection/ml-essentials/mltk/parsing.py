import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from queue import Queue, Empty
from threading import Condition, Thread, Semaphore
from typing import *

from .events import EventHost
from .typing_ import *

__all__ = [
    # program info data classes
    'ProgramInfo', 'ProgramTrainMetricInfo', 'ProgramTrainInfo',
    'ProgramWebUIInfo',

    # program output parsers
    'ProgramOutputParser', 'TFSnippetTrainOutputParser',
    'GeneralWebUIOutputParser',

    # program output receiver
    'ProgramOutputReceiver',
]


@dataclass
class ProgramInfo(object):
    """Base class for classes to store parsed program information."""


@dataclass
class ProgramTrainMetricInfo(ProgramInfo):
    __slots__ = ('mean', 'std')

    mean: Optional[str]
    std: Optional[str]


@dataclass
class ProgramTrainInfo(ProgramInfo):
    __slots__ = ('epoch', 'max_epoch', 'batch', 'max_batch',
                 'step', 'max_step', 'eta', 'epoch_eta', 'metrics')

    epoch: Optional[int]
    batch: Optional[int]
    step: Optional[int]
    max_epoch: Optional[int]
    max_batch: Optional[int]
    max_step: Optional[int]
    eta: Optional[str]
    epoch_eta: Optional[str]
    metrics: Optional[Dict[str, ProgramTrainMetricInfo]]


@dataclass
class ProgramWebUIInfo(ProgramInfo):
    __slots__ = ('name', 'uri')

    name: str
    uri: str


TInfo = TypeVar('TInfo')


class ProgramOutputParser(Generic[TInfo], metaclass=ABCMeta):
    """Base class for all program output line parsers."""

    first_n_bytes_only: Optional[int] = None
    """
    If not :obj:`None`, indicating this parser only parses this number of
    bytes at the beginning of the program output.
    """

    max_line_length: Optional[int] = None
    """
    If not :obj:`None`, indicating the maximum size of lines can be parsed
    by this parser. 
    """

    def reset(self):
        """Reset the internal states of this parser."""

    @abstractmethod
    def parse_line(self, line: bytes) -> Generator[TInfo, None, None]:
        """
        Parse one program output line.

        Args:
            line: The output line.

        Yields:
            The discovered program status objects.
        """


def re_compile_bytes(pattern: Union[bytes, str, PatternType]) -> PatternType:
    """
    Compile a bytes regex pattern.

    >>> re_compile_bytes(b'.*')
    re.compile(b'.*')
    >>> re_compile_bytes('.*')
    re.compile(b'.*')
    >>> re_compile_bytes(re.compile(b'.*', re.I))
    re.compile(b'.*', re.IGNORECASE)
    >>> re_compile_bytes(re.compile('.*'))
    Traceback (most recent call last):
        ...
    TypeError: `pattern` is not a bytes, a str or a bytes regex pattern: ...

    Args:
        pattern: The bytes regex pattern.

    Returns:
        The compiled pattern.
    """
    if not isinstance(pattern, PatternType):
        if isinstance(pattern, str):
            pattern = pattern.encode('utf-8')
        if isinstance(pattern, bytes):
            pattern = re.compile(pattern)
    if not isinstance(pattern, PatternType) or \
            not isinstance(pattern.pattern, bytes):
        raise TypeError(f'`pattern` is not a bytes, a str or a bytes regex '
                        f'pattern: got {pattern!r}')
    return pattern


class TFSnippetTrainOutputParser(ProgramOutputParser[ProgramTrainInfo]):
    """
    Training progress information parser for TFSnippet generated logs.

    >>> parser = TFSnippetTrainOutputParser()

    >>> items = list(parser.parse_line(b'[Epoch 21/100, Batch 32/99, Step 555/999, ETA 1d 3m]'
    ...                                b'loss: 0.875; acc: 0.91 (\\xc2\\xb10.01)'))
    >>> len(items)
    1
    >>> (items[0].epoch, items[0].max_epoch)
    (21, 100)
    >>> (items[0].batch, items[0].max_batch)
    (32, 99)
    >>> (items[0].step, items[0].max_step)
    (555, 999)
    >>> items[0].eta
    '1d 3m'
    >>> items[0].metrics
    {'loss': ProgramTrainMetricInfo(mean='0.875', std=None), 'acc': ProgramTrainMetricInfo(mean='0.91', std='0.01')}

    >>> list(parser.parse_line(b'[Epoch 39, Step 666]'))
    [ProgramTrainInfo(epoch=39, batch=None, step=666, max_epoch=None, max_batch=None, max_step=None, eta=None, epoch_eta=None, metrics={})]

    >>> list(parser.parse_line(b'not matched'))
    []
    """

    line_pattern = re.compile(
        rb'^\['
        rb'(?:Epoch (?P<epoch>\d+)(?:/(?P<max_epoch>\d+))?)?[, ]*'
        rb'(?:Batch (?P<batch>\d+)(?:/(?P<max_batch>\d+))?)?[, ]*'
        rb'(?:Step (?P<step>\d+)(?:/(?P<max_step>\d+))?)?[, ]*'
        rb'(?:ETA (?P<eta>[0-9\.e+ dhms]+))?'
        rb'\]\s*'
        rb'(?P<metrics>.*?)\s*(?:\(\*\))?\s*'
        rb'$'
    )
    metric_sep = re.compile(b';')
    metric_pattern = re.compile(
        rb'^\s*(?P<name>[^:]+): (?P<mean>[^()]+)'
        rb'(?: \(\xc2\xb1(?P<std>[^()]+)\))?\s*$'
    )

    def parse_line(self, line: bytes
                   ) -> Generator[ProgramTrainInfo, None, None]:
        m = self.line_pattern.match(line)
        if m:
            g = m.groupdict()

            # the progress
            info = {k: None for k in ('epoch', 'max_epoch',
                                      'batch', 'max_batch',
                                      'step', 'max_step')}
            for key in info.keys():
                if g.get(key, None) is not None:
                    info[key] = int(g[key])

            info['eta'] = info['epoch_eta'] = None
            if g.get('eta', None) is not None:
                info['eta'] = g['eta'].decode('utf-8').strip()

            # the metrics
            metrics = {}
            metric_pieces = g.pop('metrics', None)

            if metric_pieces:
                metric_pieces = self.metric_sep.split(metric_pieces)
                for metric in metric_pieces:
                    m = self.metric_pattern.match(metric)
                    if m:
                        g = m.groupdict()
                        name = g['name'].decode('utf-8').strip()
                        mean = g['mean'].decode('utf-8').strip()
                        if g.get('std', None) is not None:
                            std = g['std'].decode('utf-8').strip()
                        else:
                            std = None

                        # special hack: tfsnippet replaced "_" by " ",
                        # but we now do not use this replacement.
                        name = name.replace(' ', '_')
                        metrics[name] = (mean, std)

            # filter out none items
            metrics = {
                k: ProgramTrainMetricInfo(mean=v[0], std=v[1])
                for k, v in metrics.items()
                if v is not None
            }

            # now trigger the event
            info['metrics'] = metrics
            yield ProgramTrainInfo(**info)


class GeneralWebUIOutputParser(ProgramOutputParser[ProgramWebUIInfo]):
    """
    A general parser for WebUI information pair ``(name, uri)``.

    >>> parser = GeneralWebUIOutputParser([
    ...     re.compile(rb'^MyHTTPServer started at (?P<MyHTTPServer>\\S+)')
    ... ])

    >>> list(parser.parse_line(b'TensorBoard 1.13.1 at http://127.0.0.1:62462 '
    ...                        b'(Press CTRL+C to quit)'))
    [ProgramWebUIInfo(name='TensorBoard', uri='http://127.0.0.1:62462')]

    >>> list(parser.parse_line(b'Serving HTTP on 0.0.0.0 port 8000 '
    ...                        b'(http://0.0.0.0:8000/) ...'))
    [ProgramWebUIInfo(name='SimpleHTTP', uri='http://0.0.0.0:8000/')]

    >>> list(parser.parse_line(b'MyHTTPServer started at http://0.0.0.0:5050 ...'))
    [ProgramWebUIInfo(name='MyHTTPServer', uri='http://0.0.0.0:5050')]
    """

    BUILTIN_PATTERNS = re.compile(
        rb'(?:^TensorBoard \S+ at (?P<TensorBoard>\S+))|'
        rb'(?:^Serving HTTP on \S+ port \d+ \((?P<SimpleHTTP>[^()]+)\))'
    )

    def __init__(self, patterns: Optional[Sequence[Union[bytes,
                                                         str,
                                                         PatternType]]] = None):
        """
        Construct a new :class:`GeneralWebUIOutputParser`.

        Args:
            patterns: Additional patterns of the WebUI texts.
                These patterns will be merged with the built-in patterns.
        """
        patterns = list(map(re_compile_bytes, patterns or ()))
        self.patterns = patterns + [self.BUILTIN_PATTERNS]

    def parse_line(self, line: bytes
                   ) -> Generator[ProgramWebUIInfo, None, None]:
        for pattern in self.patterns:
            m = pattern.match(line)
            if m:
                g = m.groupdict()
                for key, val in g.items():
                    if val is not None:
                        uri = val.decode('utf-8')
                        yield ProgramWebUIInfo(name=key, uri=uri)


class ProgramOutputReceiver(object):
    """
    Receive raw program output, organizing the raw output into lines, then
    fed into the downstream :class:`ProgramOutputParser` instances.
    """

    EOF = object()
    LINE_SEP = re.compile(rb'[\r\n\b]+')

    def __init__(self,
                 parsers: Sequence[ProgramOutputParser],
                 max_line_length: int = 2048,
                 read_line_timeout: Optional[float] = 0.1,
                 max_queue_size: int = 32):
        """
        Construct a new :class:`ProgramOutputReceiver`.

        Args:
            parsers: The downstream parsers.
            max_line_length: Maximum length of each line.
            read_line_timeout: Seconds to wait before processing an incomplete
                line without seeing a link break.
            max_queue_size: The maximum size for the internal queue.
        """
        self.parsers: List[ProgramOutputParser] = list(parsers)
        self.max_line_length: int = max_line_length
        self.read_line_timeout: Optional[float] = read_line_timeout
        self.events = EventHost()
        self.on_program_info = self.events['on_program_info']
        self._queue = Queue(max_queue_size)
        self._cond = Condition()
        self._start_sem = Semaphore(0)
        self._thread: Optional[Thread] = None

    def put_output(self, data: bytes):
        """
        Put a piece of program output into the receiver.

        Args:
            data: The program output.
        """
        with self._cond:
            self._queue.put(data)

    def start(self):
        """Start the receiver in a background thread."""
        if self._thread is not None:
            raise RuntimeError('Background worker has already started.')
        self._thread = Thread(target=self._reader_func, daemon=True)
        self._thread.start()
        self._start_sem.acquire()

    def stop(self):
        """
        Stop the background receiver.

        This will force the last incomplete line to be processed.
        """
        if self._thread is not None:
            thread = self._thread
            with self._cond:
                self._queue.put(self.EOF)
            thread.join()

    def _parse_line(self, line_start: int, line: bytes):
        for parser in self.parsers:
            # check whether or not this parser can parse this line
            if parser.max_line_length is not None and \
                    len(line) > parser.max_line_length:
                continue
            if parser.first_n_bytes_only is not None and \
                    line_start >= parser.first_n_bytes_only:
                continue

            # parse the line with the selected parser
            try:
                any_info = False
                for info in parser.parse_line(line):
                    any_info = True
                    try:
                        self.on_program_info.fire(info)
                    except Exception:
                        getLogger(__name__).warning(
                            'Program info event callback error',
                            exc_info=True
                        )
                if any_info:
                    # stop parsing if a parser recognizes any program info
                    break
            except Exception:
                getLogger(__name__).warning(
                    'Program output parser failure', exc_info=True)

    def _split_lines(self,
                     parse_start: int,
                     remains: Optional[bytes],
                     data: Optional[bytes],
                     allow_remains: bool = True
                     ) -> Tuple[int, Optional[bytes]]:
        remains_length = 0 if not remains else len(remains)
        line_start = parse_start

        # `remains` must not have line separator, so we only need to find
        # the first line break in `data`
        if data:
            breaks = self.LINE_SEP.finditer(data)
            try:
                m = next(breaks)
            except StopIteration:
                # no line break is found in `data`, then merge it with the
                # old `remains`.
                if remains is not None:
                    remains = remains + data
                else:
                    remains = data
            else:
                # found the first line break in `data`, merge it with the old
                # `remains` and parse the whole line
                line = data[: m.span()[0]]
                if remains is not None:
                    line = remains + line
                if line:
                    self._parse_line(line_start, line)

                # now process the remaining lines
                start = m.span()[1]
                line_start = parse_start + remains_length + start

                for m in breaks:
                    end = m.span()[0]
                    line = data[start: end]
                    self._parse_line(line_start, line)
                    start = m.span()[1]
                    line_start = parse_start + remains_length + start

                # we're here again at the final (maybe) incomplete line,
                # use it as the new remains
                if start < len(data):
                    remains = data[start:]
                else:
                    remains = None

        # if there is remains, we parse it as a whole line if `allow_remains`
        # is :obj:`False`, or if `len(remains)` > max_line_length
        if remains:
            if not allow_remains or len(remains) >= self.max_line_length:
                self._parse_line(line_start, remains)
                line_start += len(remains)
                remains = None

        return line_start, remains

    def _reader_func(self):
        running: bool = True
        remains: Optional[bytes] = None
        parse_start: int = 0
        for parser in self.parsers:
            parser.reset()
        self._start_sem.release()

        while running:
            # get the next starting chunk, waiting forever
            itm = self._queue.get()
            if itm is self.EOF:
                break
            parse_start, remains = self._split_lines(parse_start, remains, itm)

            # If `read_line_timeout` is specified, once we've got any non-empty
            # but incomplete line, we shall wait for at most `read_line_timeout`
            # seconds of time for the line to be completed before we parsing
            # the incomplete line as a whole line.
            if remains and self.read_line_timeout is not None:
                while True:
                    try:
                        itm = self._queue.get(timeout=self.read_line_timeout)
                        if itm is self.EOF:
                            running = False
                            break
                        else:
                            parse_start, remains = \
                                self._split_lines(parse_start, remains, itm)
                    except Empty:
                        break

                # Once timeout has occurred in this situation, we need
                # to parse the incomplete line as a whole line immediately.
                if remains:
                    parse_start, remains = \
                        self._split_lines(parse_start, remains, None, False)

        # If there is still remains, then the reader loop should have been
        # interrupted by EOF.  Thus we need to parse the remaining incomplete
        # line immediately.
        if remains:
            self._split_lines(parse_start, remains, None, False)

        # Thread will exit, de-reference this thread
        self._thread = None
