import os
import re
import shutil
import sys
import unittest
from tempfile import TemporaryDirectory

import pytest

from mltk.utils import *
from tests.helpers import prepare_dir, zip_snapshot


class MakeDirArchiveTestCase(unittest.TestCase):

    def test_make_dir_archive(self):
        with TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, 'source')
            archive_path = os.path.join(temp_dir, 'archive.zip')

            with pytest.raises(IOError, match=f'Not a directory: {source_dir}'):
                make_dir_archive(source_dir, archive_path)

            # prepare for the source dir
            source_content = {
                'a.txt': b'a.txt',
                'a.exe': b'a.exe',
                'nested': {
                    'b.py': b'b.py',
                    '.DS_Store': b'.DS_Store',
                },
                '.git': {
                    'c.sh': b'c.sh',
                }
            }
            prepare_dir(source_dir, source_content)

            # test make archive without filters
            make_dir_archive(source_dir, archive_path)
            self.assertDictEqual(zip_snapshot(archive_path), source_content)

            # test make archive with filters
            archive_path = os.path.join(temp_dir, 'archive.zip')
            make_dir_archive(
                source_dir,
                archive_path,
                is_included=re.compile(r'.*(\.(txt|py|sh)|\.DS_Store)$').match,
                is_excluded=re.compile(r'.*(\.git|\.DS_Store)$').match,
            )
            self.assertDictEqual(zip_snapshot(archive_path), {
                'a.txt': b'a.txt',
                'nested': {
                    'b.py': b'b.py',
                }
            })


class ExtractorTestCase(unittest.TestCase):

    def check_archive_file(self, extractor_class, archive_file, alias=None):
        if alias is not None:
            with TemporaryDirectory() as tmpdir:
                new_archive_file = os.path.join(tmpdir, alias)
                shutil.copy(archive_file, new_archive_file)
                self.check_archive_file(extractor_class, new_archive_file)
        else:
            with Extractor.open(archive_file) as e:
                self.assertIsInstance(e, extractor_class)
                files = [(n, f.read()) for n, f in e.iter_extract()]
                self.assertListEqual(
                    [
                        ('a/1.txt', b'a/1.txt'),
                        ('b/2.txt', b'b/2.txt'),
                        ('c.txt', b'c.txt'),
                    ],
                    files
                )

    def get_asset(self, name):
        return os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            '../assets/utils',
            name
        )

    def test_zip(self):
        self.check_archive_file(ZipExtractor, self.get_asset('payload.zip'))

    def test_rar(self):
        self.check_archive_file(RarExtractor, self.get_asset('payload.rar'))

    def test_tar(self):
        self.check_archive_file(TarExtractor, self.get_asset('payload.tar'))
        # xz
        if sys.version_info[:2] >= (3, 3):
            self.check_archive_file(
                TarExtractor, self.get_asset('payload.tar.xz'))
            self.check_archive_file(
                TarExtractor, self.get_asset('payload.tar.xz'), 'payload.txz')
        # gz
        self.check_archive_file(TarExtractor, self.get_asset('payload.tar.gz'))
        self.check_archive_file(TarExtractor, self.get_asset('payload.tar.gz'),
                                'payload.tgz')
        # bz2
        self.check_archive_file(TarExtractor, self.get_asset('payload.tar.bz2'))
        self.check_archive_file(TarExtractor, self.get_asset('payload.tar.bz2'),
                                'payload.tbz')
        self.check_archive_file(TarExtractor, self.get_asset('payload.tar.bz2'),
                                'payload.tbz2')
        self.check_archive_file(TarExtractor, self.get_asset('payload.tar.bz2'),
                                'payload.tb2')

    def test_errors(self):
        with TemporaryDirectory() as tmpdir:
            archive_file = os.path.join(tmpdir, 'payload.txt')
            with open(archive_file, 'wb') as f:
                f.write(b'')
            with pytest.raises(
                    IOError, match='File is not a supported archive file'):
                with Extractor.open(archive_file):
                    pass
