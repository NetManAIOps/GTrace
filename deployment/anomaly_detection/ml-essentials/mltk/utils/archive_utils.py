import os
import tarfile
import zipfile
from typing import *

try:
    import rarfile
#    rarfile.PATH_SEP = '/'
except ImportError:  # pragma: no cover
    rarfile = None

__all__ = [
    'make_dir_archive',
    'Extractor', 'TarExtractor', 'ZipExtractor', 'RarExtractor',
]


TAR_FILE_EXTENSIONS = ('.tar',
                       '.tar.gz', '.tgz',
                       '.tar.bz2', '.tbz', '.tbz2', '.tb2',
                       '.tar.xz', '.txz')


def make_dir_archive(dir_path: str,
                     archive_path: str,
                     is_included: Callable[[str], bool] = None,
                     is_excluded: Callable[[str], bool] = None,
                     compression: int = zipfile.ZIP_STORED):
    """
    Pack all content of a directory into an archive.

    Args:
        dir_path: The source directory ptah.
        archive_path: The destination archive path.
        is_included: A callable ``(file_path) -> bool`` to check whether or not
            a file should be included in the archive.
        is_excluded: A callable ``(path) -> bool`` to check whether or not
            a directory or a file should be excluded in the archive.
        compression: The compression level.
    """
    if not os.path.isdir(dir_path):
        raise IOError(f'Not a directory: {dir_path}')

    def walk(path, relpath):
        for name in os.listdir(path):
            f_path = os.path.join(path, name)
            f_relpath = f'{relpath}/{name}' if relpath else name

            if is_excluded is not None and is_excluded(f_path):
                continue

            if os.path.isdir(f_path):
                zf.write(f_path, arcname=f_relpath)
                walk(f_path, f_relpath)
            elif is_included is None or is_included(f_path):
                zf.write(f_path, arcname=f_relpath)

    with zipfile.ZipFile(archive_path, 'w', compression=compression) as zf:
        walk(dir_path, '')


class Extractor(object):
    """
    The base class for all archive extractors.

    .. code-block:: python

        from mltk.utils import Extractor, maybe_close

        with Extractor.open('a.zip') as archive_file:
            for name, f in archive_file:
                with maybe_close(f):  # This file object may not be closeable,
                                      # thus we wrap it by ``maybe_close()``
                    print(f'the content of {name} is:')
                    print(f.read())
    """

    def __init__(self, archive_file):
        """
        Initialize the base :class:`Extractor` class.

        Args:
            archive_file: The archive file object.
        """
        self._archive_file = archive_file

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __iter__(self):
        return self.iter_extract()

    def close(self):
        """Close the extractor."""
        if self._archive_file:
            self._archive_file.close()
            self._archive_file = None

    def iter_extract(self) -> Generator[Tuple[str, Any], None, None]:
        """
        Extract files from the archive with an iterator.

        You may simply iterate over a :class:`Extractor` object, which is
        equivalent to calling this method.

        Yields:
            Tuples of ``(name, file-like object)``, the filename and
            corresponding file-like object for each file in the archive.
            The returned file-like object may or may not be closeable.
            You may wrap it by :func:`mltk.utils.maybe_close()` context.
        """
        raise NotImplementedError()

    @staticmethod
    def open(file_path: str) -> 'Extractor':
        """
        Create an :class:`Extractor` instance for given archive file.

        Args:
            file_path: The path of the archive file.

        Returns:
            The specified extractor instance.

        Raises:
            IOError: If the ``file_path`` is not a supported archive.
        """
        if file_path.endswith('.rar') and rarfile is not None:
            return RarExtractor(file_path)
        elif file_path.endswith('.zip'):
            return ZipExtractor(file_path)
        elif any(file_path.endswith(ext) for ext in TAR_FILE_EXTENSIONS):
            return TarExtractor(file_path)
        else:
            raise IOError('File is not a supported archive file: {!r}'.
                          format(file_path))


def normalize_archive_entry_name(name: str) -> str:
    """
    Get the normalized name of an archive file entry.

    >>> normalize_archive_entry_name('a\\\\b/c')
    'a/b/c'

    Args:
        name: Name of the archive file entry.

    Returns:
        str: The normalized name.
    """
    return name.replace('\\', '/')


class TarExtractor(Extractor):
    """
    Extractor for ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2",
    ".tb2", ".tar.xz", ".txz" files.
    """

    def __init__(self, fpath: str):
        super(TarExtractor, self).__init__(tarfile.open(fpath, 'r'))

    def iter_extract(self) -> Generator[Tuple[str, Any], None, None]:
        for mi in self._archive_file:
            if not mi.isdir():
                yield (
                    normalize_archive_entry_name(mi.name),
                    self._archive_file.extractfile(mi)
                )


class ZipExtractor(Extractor):
    """Extractor for ".zip" files."""

    def __init__(self, fpath: str):
        super(ZipExtractor, self).__init__(zipfile.ZipFile(fpath, 'r'))

    def iter_extract(self) -> Generator[Tuple[str, Any], None, None]:
        for mi in self._archive_file.infolist():
            # ignore directory entries
            if mi.filename[-1] == '/':
                continue
            yield (
                normalize_archive_entry_name(mi.filename),
                self._archive_file.open(mi)
            )


class RarExtractor(Extractor):
    """Extractor for ".rar" files."""

    def __init__(self, fpath: str):
        if rarfile is None:  # pragma: no cover
            raise RuntimeError('Required package not installed: rarfile')
        super(RarExtractor, self).__init__(rarfile.RarFile(fpath, 'r'))

    def iter_extract(self) -> Generator[Tuple[str, Any], None, None]:
        for mi in self._archive_file.infolist():
            if mi.isdir():
                continue
            yield (
                normalize_archive_entry_name(mi.filename),
                self._archive_file.open(mi)
            )
