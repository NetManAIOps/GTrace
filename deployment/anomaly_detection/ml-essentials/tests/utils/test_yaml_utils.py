import codecs
import os
import unittest
import warnings
from tempfile import TemporaryDirectory

import pytest
import yaml

from mltk.utils import *


class YAMLUtilsTestCase(unittest.TestCase):

    def test_include_loader(self):
        with TemporaryDirectory() as temp_dir:
            file_name = os.path.join(temp_dir, 'x.yml')
            file_name2 = os.path.join(temp_dir, 'y.yml')

            with codecs.open(file_name, 'wb', 'utf-8') as f:
                f.write('value: 123\n')
            with codecs.open(file_name2, 'wb', 'utf-8') as f:
                f.write('!include x.yml\n')

            # test load with relative path according to f.name
            with codecs.open(file_name2, 'rb', 'utf-8') as f:
                obj = yaml.load(f, Loader=YAMLIncludeLoader)
            self.assertDictEqual(obj, {'value': 123})

            # test load with relative path according to root_dir
            with codecs.open(file_name2, 'rb', 'utf-8') as f:
                obj = YAMLIncludeLoader(f.read(), root_dir=temp_dir).get_single_data()
            self.assertDictEqual(obj, {'value': 123})

            # test load with absolute path
            obj = yaml.load(f'!include "{file_name}"\n', Loader=YAMLIncludeLoader)
            self.assertDictEqual(obj, {'value': 123})
