import os

from .config import Config, config_field

__all__ = ['Settings', 'settings']


class Settings(Config):
    """Global settings of the whole `mltk` package."""

    cache_root: str = config_field(
        default=os.path.expanduser('~/.mltk/cache'),
        envvar='MLTK_CACHE_ROOT',
    )

    file_cache_checksum: bool = False
    """Whether or not to validate the checksum of cached files?"""


settings = Settings()
