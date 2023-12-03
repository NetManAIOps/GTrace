from typing import *

__all__ = ['Event', 'EventHost']


class Event(object):
    """
    Event object, should only be constructed by :class:`EventHost`.
    """

    def __init__(self, host: 'EventHost', name: str):
        self._host = host
        self._name = name
        self._callbacks = []

    def __repr__(self):
        return f'Event({self._name})'

    def do(self, callback: Callable[..., None]):
        """
        Register `callback` to this event.

        Args:
            callback: Callback to register.
        """
        self._callbacks.append(callback)
        return callback

    def cancel_do(self, callback: Callable[..., None]):
        """
        Unregister `callback` from this event.

        Args:
            callback: Callback to unregister.
                No error will be raised if it has not been registered yet.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def fire(self, *args, **kwargs):
        """
        Fire this event.

        Args:
            *args: Positional arguments.
            **kwargs: Named arguments.
        """
        self._host.fire(self._name, *args, **kwargs)

    def reverse_fire(self, *args, **kwargs):
        """
        Fire this event, calling all callbacks in reversed order.

        Args:
            *args: Positional arguments.
            **kwargs: Named arguments.
        """
        self._host.reverse_fire(self._name, *args, **kwargs)


class EventHost(object):
    """
    Class to create and manage :class:`Event` objects.

    Callbacks can be registered to a event by :meth:`on()`, for example:

    >>> def print_args(*args, **kwargs):
    ...     print(args, kwargs)

    >>> events = EventHost()
    >>> events.on('updated', print_args)  # register `print_args` to a event
    >>> list(events)  # get names of created events
    ['updated']
    >>> 'updated' in events  # test whether a event name exists
    True
    >>> events.fire('updated', 123, second=456)  # fire the event
    (123,) {'second': 456}

    It is also possible to obtain an object that represents the event,
    and register callbacks / fire the event via that object, for example:

    >>> events = EventHost()
    >>> on_updated = events['updated']
    >>> list(events)
    ['updated']
    >>> on_updated
    Event(updated)
    >>> on_updated.do(print_args)
    <function print_args at ...>
    >>> on_updated.fire(123, second=456)
    (123,) {'second': 456}
    """

    def __init__(self):
        self._connected_hosts = []
        self._events = {}  # type: Dict[str, Event]

    def __iter__(self) -> Iterator[str]:
        return iter(self._events)

    def __contains__(self, item) -> bool:
        return item in self._events

    def __getitem__(self, item) -> Event:
        if item not in self._events:
            self._events[item] = Event(self, item)
        return self._events[item]

    def connect(self, other: 'EventHost'):
        """
        Connect this event host with another event host or another object,
        such that all events fired from this host will also be fired from
        that host, or the function of that object with the name of the event
        will also be called.

        Connect an event host with another event host:

        >>> events1 = EventHost()
        >>> events1.on('updated',
        ...     lambda *args, **kwargs: print('from events1', args, kwargs))
        >>> events2 = EventHost()
        >>> events2.on('updated',
        ...     lambda *args, **kwargs: print('from events2', args, kwargs))
        >>> events1.connect(events2)
        >>> events1.fire('updated', 123, second=456)
        from events1 (123,) {'second': 456}
        from events2 (123,) {'second': 456}

        Connect an event host with another object:

        >>> class MyObject(object):
        ...     def updated(self, *args, **kwargs):
        ...         print('from MyObject', args, kwargs)

        >>> events = EventHost()
        >>> obj = MyObject()
        >>> events.connect(obj)
        >>> events.fire('updated', 123, second=456)
        from MyObject (123,) {'second': 456}

        Note if a event is fired, but the connected object does not have
        a method with that name, no error will be raised, and the object
        will be silently ignored on the event propagation chain.

        Args:
            other: The other event host.
        """
        self._connected_hosts.append(other)

    def disconnect(self, other: 'EventHost'):
        """
        Disconnect this event host with another event host.

        Args:
            other: The other event host.

        See Also:
            :meth:`connect()`
        """
        if other in self._connected_hosts:
            self._connected_hosts.remove(other)

    def on(self, name: str, callback: Callable):
        """
        Register `callback` to an event.
        Args:
            name: Name of the event.
            callback: Callback to register.
        """
        self[name].do(callback)

    def off(self, name: str, callback: Callable):
        """
        Unregister `callback` from an event.

        Args:
            name: Name of the event.
            callback: Callback to unregister.
                No error will be raised if it has not been registered yet.
        """
        if name in self._events:
            self._events[name].cancel_do(callback)

    def _fire_host_event(self, host, name, reversed_, args, kwargs):
        if isinstance(host, EventHost):
            if reversed_:
                host.reverse_fire(name, *args, **kwargs)
            else:
                host.fire(name, *args, **kwargs)
        else:
            fn = getattr(host, name, None)
            if isinstance(fn, Event):
                fn.fire(*args, **kwargs)
            elif fn is not None:
                fn(*args, **kwargs)

    def fire(self, name_, *args, **kwargs):
        """
        Fire an event.

        Args:
            name_: Name of the event.
            *args: Positional arguments.
            \\**kwargs: Named arguments.
        """
        event = self._events.get(name_, None)
        if event is not None:
            for callback in event._callbacks:
                callback(*args, **kwargs)

        for host in self._connected_hosts:
            self._fire_host_event(host, name_, False, args, kwargs)

    def reverse_fire(self, name_, *args, **kwargs):
        """
        Fire an event, calling all callbacks in reversed order.

        Args:
            name_: Name of the event.
            *args: Positional arguments.
            \\**kwargs: Named arguments.
        """
        for host in reversed(self._connected_hosts):
            self._fire_host_event(host, name_, True, args, kwargs)

        event = self._events.get(name_, None)
        if event is not None:
            for callback in reversed(event._callbacks):
                callback(*args, **kwargs)
