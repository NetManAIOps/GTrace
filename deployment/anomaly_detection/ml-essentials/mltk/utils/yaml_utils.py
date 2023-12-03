import os
from typing import *

import yaml

__all__ = ['YAMLIncludeLoader']


class YAMLIncludeLoader(yaml.SafeLoader):
    """A subclass of `yaml.SafeLoader` that supports `!include` directive."""

    root_dir: Optional[str] = None

    def __init__(self, stream, root_dir: Optional[str] = None):
        if root_dir is None:
            file_name = getattr(stream, 'name', None)
            if file_name is not None:
                root_dir = os.path.split(file_name)[0]
        self.root_dir = root_dir

        super().__init__(stream)

    def include(self, node):
        filename = self.construct_scalar(node)
        if self.root_dir is not None:
            filename = os.path.join(self.root_dir, filename)
        with open(filename, 'r') as f:
            return yaml.load(f, yaml.SafeLoader)


YAMLIncludeLoader.add_constructor('!include', YAMLIncludeLoader.include)
