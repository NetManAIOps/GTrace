import warnings
from functools import wraps
from typing import *

__all__ = ['deprecated_arg']


def make_deprecated_warning(message: str):
    """
    Generate a deprecation warning.

    Args:
        message: The deprecation message.
    """
    warnings.warn(message, DeprecationWarning)


def deprecated_arg(name: str,
                   new_name: Optional[str] = None,
                   message: Optional[str] = None):
    """
    Decorator that marks an argument to be deprecated.

    The argument must be a keyword-only argument, otherwise this decorator
    may not be able to capture the argument value.

    Args:
        name: Name of the argument.
        new_name: If specified, will automatically remove the deprecated
            keyword argument from `\\**kwargs`, and add a new keyword
            argument with this name into `\\**kwargs`.  The new name must
            not appear in `\\**kwargs` yet.
        message: Additional warning message.

    Returns:
        The decorated method.
    """
    message = f': {message}' or ''

    def wrapper(method):
        @wraps(method)
        def inner(*args, **kwargs):
            if name in kwargs:
                if new_name is None:
                    make_deprecated_warning(
                        f'The argument `{name}` is deprecated{message}.')
                else:
                    make_deprecated_warning(
                        f'The argument `{name}` is deprecated, use '
                        f'`{new_name}` instead{message}.')
                    if new_name in kwargs:
                        raise ValueError(
                            f'Values are specified to both the deprecated '
                            f'argument `{name}` and the new argument '
                            f'`{new_name}` to replace it.'
                        )
                    kwargs[new_name] = kwargs.pop(name)
            return method(*args, **kwargs)
        return inner
    return wrapper
