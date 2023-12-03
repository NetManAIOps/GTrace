__all__ = [
    'UserTermination', 'BaseLoopError', 'NaNMetricError',
]


class UserTermination(Exception):
    """
    Exception that forces a loop to be terminated.

    Different from the :meth:`Stage.request_termination()`, which can only
    be handled by :class:`mltk.loop.BaseLoop`, this exception will cause
    any loop to be interrupted immediately in all situations.
    :class:`mltk.callbacks.EarlyStopping` will always ignore this error, and
    restore the best saved checkpoint.
    One drawback is that the user must catch this exception outside the loop.
    """


class BaseLoopError(Exception):
    """
    Base class for errors occurred in a train/validation/test/predict loop.
    """


class NaNMetricError(BaseLoopError):
    """Error that indicates an NaN metric has been encountered."""

    def __init__(self, metric_name: str):
        super().__init__(metric_name)

    @property
    def metric_name(self) -> str:
        return self.args[0]

    def __str__(self):
        return f'NaN metric encountered: {self.metric_name!r}'
