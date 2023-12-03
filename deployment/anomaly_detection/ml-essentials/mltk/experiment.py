import codecs
import os
import shutil
import sys
import time
import zipfile
from argparse import ArgumentParser
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import *

from bson import ObjectId

from .config import Config, ConfigLoader, config_to_dict, config_defaults
from .events import EventHost, Event
from .formatting import format_as_asctime
from .mlstorage import MLStorageClient, ExperimentDoc
from .utils import (NOT_SET, make_dir_archive, json_dumps, json_loads,
                    ContextStack, deprecated_arg)
from .utils.remote_doc import merge_updates_into_doc
from .type_check import DiscardMode
from .typing_ import TConfig

__all__ = ['Experiment', 'get_active_experiment']


class Experiment(Generic[TConfig]):
    """
    Class to manage the configuration and results of an experiment.

    Basic Usage
    ===========

    To use this class, you first define your experiment config with
    :class:`mltk.Config`, and then wrap your main experiment routine with an
    :class:`Experiment` context, for example, you may write your `main.py` as::

        import numpy as np
        from mltk import Config, Experiment


        class YourConfig(Config):
            max_epoch: int = 100
            learning_rate: float = 0.001
            ...

        if __name__ == '__main__':
            with Experiment(YourConfig) as exp:
                # `exp.config` is the configuration values
                print('Max epoch: ', exp.config.max_epoch)
                print('Learning rate: ', exp.config.learning_rate)

                # `exp.output_dir` is the output directory of this experiment
                print('Output directory: ', exp.output_dir)

                # save result metrics into `exp.output_dir + "/result.json"`
                exp.update_results({'test_acc': ...})

                # write result arrays into `exp.output_dir + "/data.npz"`
                output_file = exp.abspath('data.npz')
                np.savez(output_file, predict=...)

    Then you may execute your python file via::

        python main.py  # use the default config to run the file
        python main.py --max_epoch=200  # override some of the config

    The output directory (i.e., `exp.output_dir`) will be by default
    `"./results/" + script_name`, where `script_name` is the file name of the
    main module (excluding ".py").  The configuration values will be saved
    into `exp.output_dir + "/config.json"`.  If the config file already exists,
    it will be loaded and merged with the config values specified by the CLI
    arguments.  This behavior can allow resuming from an interrupted experiment.

    If the Python script is executed via `mlrun`, then the output directory
    will be assigned by MLStorage server.  For example::

        mlrun -s http://<server>:<port> -- python main.py --max_epoch=200

    You may also get the server URI and the experiment ID assigned by the
    server via the properties `id` and `client`.

    To resume from an interrupted experiment with `mlrun`::

        mlrun --resume-from=<experiment id> -- python main.py

    Since `mlrun` will pick up the output directory of the previous experiment,
    :class:`Experiment` will correctly restore the configuration values from
    `exp.output_dir + "/config.json"`, thus no need to specify the CLI arguments
    once again when resuming from an experiment.
    """

    def __init__(self,
                 config_or_cls: Union[Type[TConfig], TConfig],
                 *,
                 script_name: str = NOT_SET,
                 output_dir: Optional[str] = NOT_SET,
                 args: Optional[Sequence[str]] = NOT_SET,
                 auto_load_config: bool = True,
                 auto_save_config: bool = True,
                 discard_undefind_config_fields: Union[str, DiscardMode] = DiscardMode.WARN,
                 ):
        """
        Construct a new :class:`Experiment`.

        Args:
            config_or_cls: The configuration object, or class.
            script_name: The script name.  By default use the file name
                of ``sys.modules['__main__']`` (excluding ".py").
            output_dir: The output directory.  If not specified, use
                `"./results/" + script_name`, or assigned by MLStorage
                server if the experiment is launched by `mlrun`.

                Set to :obj:`None` will disable using an output directory.
                In this case, any method that attempt to operate on the
                output directory will cause an error.
            args: The CLI arguments.  If not specified, use ``sys.argv[1:]``.
                Specifying :obj:`None` will disable parsing the arguments.
            auto_load_config: Whether or not to restore configuration
                values from the previously saved `output_dir + "/config.json"`?
                If `output_dir` is None, this argument will be ignored.
            auto_save_config: Whether or not to save configuration
                values into `output_dir + "/config.json"`?
                If `output_dir` is None, this argument will be ignored.
            discard_undefind_config_fields: The mode to deal with undefined
                config fields when loading from previously saved config file.
                Defaults to ``DiscardMode.WARN``, where the undefined config
                fields are automatically discarded, with a warning generated.
        """
        # validate the arguments
        config_or_cls_okay = True
        config = None

        if isinstance(config_or_cls, type):
            if not issubclass(config_or_cls, Config):
                config_or_cls_okay = False
            else:
                config = config_or_cls()
        else:
            if not isinstance(config_or_cls, Config):
                config_or_cls_okay = False
            else:
                config = config_or_cls

        if not config_or_cls_okay:
            raise TypeError(f'`config_or_cls` is neither a Config class, '
                            f'nor a Config instance: {config_or_cls!r}')

        if script_name is NOT_SET:
            script_name = os.path.splitext(
                os.path.basename(sys.modules['__main__'].__file__))[0]

        if output_dir is NOT_SET:
            output_dir = os.environ.get('MLSTORAGE_OUTPUT_DIR', NOT_SET)
        if output_dir is NOT_SET:
            candidate_dir = f'./results/{script_name}'
            while os.path.exists(candidate_dir):
                time.sleep(0.01)
                suffix = format_as_asctime(
                    datetime.now(),
                    datetime_format='%Y-%m-%d_%H-%M-%S',
                    datetime_msec_sep='_',
                )
                candidate_dir = f'./results/{script_name}_{suffix}'
            output_dir = candidate_dir
        if output_dir is not None:
            output_dir = os.path.abspath(output_dir)

        if args is NOT_SET:
            args = sys.argv[1:]
        if args is not None:
            args = tuple(map(str, args))

        # memorize the arguments
        self._config = config
        self._script_name = script_name
        self._output_dir = output_dir
        self._args = args
        self._auto_load_config = auto_load_config
        self._auto_save_config = auto_save_config
        self._discard_undefind_config_fields = discard_undefind_config_fields

        # the event
        self._events = EventHost()
        self._on_enter = self.events['on_enter']
        self._on_exit = self.events['on_exit']

        # initialize the MLStorage client if environment variable is set
        id = os.environ.get('MLSTORAGE_EXPERIMENT_ID', None) or None
        if id is not None:
            id = ObjectId(id)
        if os.environ.get('MLSTORAGE_SERVER_URI', None):
            client = MLStorageClient(os.environ['MLSTORAGE_SERVER_URI'])
        else:
            client = None

        self._id = id
        self._client = client

        # construct the experiment document
        self._remote_doc = ExperimentDoc(client=client, id=id)
        self._remote_doc.on_before_push.do(self._on_before_push_doc)

    @property
    def id(self) -> Optional[str]:
        """Get the experiment ID, if the environment variable is set."""
        return self._id

    @property
    def client(self) -> Optional[MLStorageClient]:
        """Get the MLStorage client, if the environment variable is set."""
        return self._client

    @property
    def doc(self) -> ExperimentDoc:
        """Get the experiment document object."""
        return self._remote_doc

    @property
    def config(self) -> TConfig:
        """
        Get the config object.

        If you would like to modify this object, you may need to manually call
        :meth:`save_config()`, in order to save the modifications to disk.
        """
        return self._config

    @property
    def results(self) -> Optional[Mapping[str, Any]]:
        """
        Get the results, or :obj:`None` if no result has been written.
        """
        return self.doc.get('result', None)

    @property
    def script_name(self) -> str:
        """Get the script name of this experiment."""
        return self._script_name

    @property
    def output_dir(self) -> Optional[str]:
        """
        Get the output directory of this experiment.

        Returns:
            The output directory, or :obj:`None` if the experiment object
            does not have an output directory.
        """
        return self._output_dir

    @property
    def args(self) -> Optional[Tuple[str]]:
        """Get the CLI arguments of this experiment."""
        return self._args

    @property
    def events(self) -> EventHost:
        """Get the event host."""
        return self._events

    @property
    def on_enter(self) -> Event:
        """
        Get the on enter event.

        Callback function type: `() -> None`

        This event will be triggered when entering an experiment context::

            with Experiment(...) as exp:
                # this event will be triggered after entering the context,
                # and before the following statements

                ...
        """
        return self._on_enter

    @property
    def on_exit(self) -> Event:
        """
        Get the on exit event.

        Callback function type: `() -> None`

        This event will be triggered when exiting an experiment context::

            with Experiment(...) as exp:
                ...

                # this event will be triggered after the above statements,
                # and before exiting the context
        """
        return self._on_exit

    def _require_output_dir(self):
        if self._output_dir is None:
            raise RuntimeError('No output directory is configured.')

    def _on_before_push_doc(self, updates: Dict[str, Any]):
        """Callback to save the updated fields to local JSON file."""

        # gather updated fields of: 'config', 'default_config', 'result'
        updated_fields = {}
        merge_updates_into_doc(
            updated_fields,
            updates,
            keys_to_expand=('config', 'default_config', 'result'))

        # if no output dir, do not save anything
        if self._output_dir is None:
            return

        # now save the interested fields
        def save_json_file(path, obj, merge: bool = False):
            if merge:
                try:
                    if os.path.exists(path):
                        with codecs.open(path, 'rb', 'utf-8') as f:
                            old_obj = json_loads(f.read())
                        obj = merge_updates_into_doc(old_obj, obj)
                except Exception:  # pragma: no cover
                    raise IOError(f'Cannot load the existing JSON file: {path}')
            obj_json = json_dumps(obj, no_dollar_field=True)
            with codecs.open(path, 'wb', 'utf-8') as f:
                f.write(obj_json)

        # NOTE: Do not makedirs until we've confirmed that some fields need
        #       to be saved.
        if any(key in updated_fields
               for key in ('config', 'default_config', 'result')):
            os.makedirs(self.output_dir, exist_ok=True)

        if 'config' in updated_fields:
            save_json_file(os.path.join(self.output_dir, 'config.json'),
                           updated_fields['config'])
        if 'default_config' in updated_fields:
            save_json_file(os.path.join(self.output_dir, 'config.defaults.json'),
                           updated_fields['default_config'])
        if 'result' in updated_fields:
            save_json_file(os.path.join(self.output_dir, 'result.json'),
                           updated_fields['result'], merge=True)

    def save_config(self):
        """Save the config into local file and to remote server."""
        self._remote_doc.update(
            {
                'config': config_to_dict(self.config, flatten=True),
                'default_config': config_to_dict(
                    config_defaults(self.config), flatten=True)
            },
            immediately=True
        )

    def update_results(self,
                       results: Optional[Dict[str, Any]] = None,
                       **kwargs):
        """
        Update the result dict.

        Args:
            results: The dict of updates.
            **kwargs: The named arguments of updates.
        """
        results = dict(results or ())
        results.update(kwargs)
        self._remote_doc.update({'result': results})

    def abspath(self, relpath: str) -> str:
        """
        Get the absolute path of a relative path in `output_dir`.

        Args:
            relpath: The relative path.

        Returns:
            The absolute path of `relpath`.
        """
        self._require_output_dir()
        return os.path.join(self.output_dir, relpath)

    def make_dirs(self, relpath: str, exist_ok: bool = True) -> str:
        """
        Create a directory (and its ancestors if necessary) in `output_dir`.

        Args:
            relpath: The relative path of the directory.
            exist_ok: If :obj:`True`, will not raise error if the directory
                already exists.

        Returns:
            The absolute path of `relpath`.
        """
        self._require_output_dir()
        path = self.abspath(relpath)
        os.makedirs(path, exist_ok=exist_ok)
        return path

    def make_parent(self, relpath: str, exist_ok: bool = True) -> str:
        """
        Create the parent directory of `relpath` (and its ancestors if
        necessary) in `output_dir`.

        Args:
            relpath: The relative path of the entry, whose parent and
                ancestors are to be created.
            exist_ok: If :obj:`True`, will not raise error if the parent
                directory already exists.

        Returns:
            The absolute path of `relpath`.
        """
        self._require_output_dir()
        path = self.abspath(relpath)
        parent_dir = os.path.split(path)[0]
        os.makedirs(parent_dir, exist_ok=exist_ok)
        return path

    def open_file(self, relpath: str, mode: str, encoding: Optional[str] = None,
                  make_parent: bool = NOT_SET):
        """
        Open a file at `relpath` in `output_dir`.

        Args:
            relpath: The relative path of the file.
            mode: The open mode.
            encoding: The text encoding.  If not specified, will open the file
                in binary mode; otherwise will open it in text mode.
            make_parent: If :obj:`True`, will create the parent (and all
                ancestors) of `relpath` if necessary.  By default, will
                create the parent if open the file by writable mode.

        Returns:
            The opened file.
        """
        self._require_output_dir()
        if make_parent is NOT_SET:
            make_parent = any(s in mode for s in 'aw+')

        if make_parent:
            path = self.make_parent(relpath)
        else:
            path = self.abspath(relpath)

        if encoding is None:
            return open(path, mode)
        else:
            return codecs.open(path, mode, encoding)

    def put_file_content(self,
                         relpath: str,
                         content: Union[bytes, str],
                         append: bool = False,
                         encoding: Optional[str] = None):
        """
        Save content into a file.

        Args:
            relpath: The relative path of the file.
            content: The file content.  Must be bytes if `encoding` is not
                specified, while text if `encoding` is specified.
            append: Whether or not to append to the file?
            encoding: The text encoding.
        """
        self._require_output_dir()
        with self.open_file(relpath, 'ab' if append else 'wb',
                            encoding=encoding) as f:
            f.write(content)

    def get_file_content(self, relpath: str, encoding: Optional[str] = None
                         ) -> Union[bytes, str]:
        """
        Get the content of a file.

        Args:
            relpath: The relative path of a file.
            encoding: The text encoding.  If specified, will decode the
                file content using this encoding.

        Returns:
            The file content.
        """
        self._require_output_dir()
        with self.open_file(relpath, 'rb', encoding=encoding) as f:
            return f.read()

    def make_archive(self,
                     source_dir: str,
                     archive_file: Optional[str] = None,
                     delete_source: bool = True):
        """
        Pack a directory into a zip archive.

        For repeated experiments, pack some result directories into zip
        archives will reduce the total inode count of the file system.

        Args:
            source_dir: The relative path of the source directory.
            archive_file: The relative path of the zip archive.
                If not specified, will use `source_dir + ".zip"`.
            delete_source: Whether or not to delete `source_dir` after
                the zip archive has been created?

        Returns:
            The absolute path of the archive file.
        """
        self._require_output_dir()

        def _copy_dir(src: str, dst: str):
            os.makedirs(dst, exist_ok=True)

            for name in os.listdir(src):
                f_src = os.path.join(src, name)
                f_dst = os.path.join(dst, name)

                if os.path.isdir(f_src):
                    _copy_dir(f_src, f_dst)
                else:
                    shutil.copyfile(f_src, f_dst, follow_symlinks=False)

        source_dir = self.abspath(source_dir)
        if not os.path.isdir(source_dir):
            raise IOError(f'Not a directory: {source_dir}')

        if archive_file is None:
            archive_file = source_dir.rstrip('/\\') + '.zip'
        else:
            archive_file = self.abspath(archive_file)

        def prepare_parent():
            parent_dir = os.path.dirname(archive_file)
            os.makedirs(parent_dir, exist_ok=True)

        # if the archive already exists, extract it, merge the contents
        # in `source_dir` with the extracted files, and then make archive.
        if os.path.isfile(archive_file):
            with TemporaryDirectory() as temp_dir:
                # extract the original zip
                with zipfile.ZipFile(archive_file, 'r') as zf:
                    zf.extractall(temp_dir)

                # merge the content
                _copy_dir(source_dir, temp_dir)

                # make the destination archive
                prepare_parent()
                make_dir_archive(temp_dir, archive_file)

        # otherwise pack the zip archive directly
        else:
            prepare_parent()
            make_dir_archive(source_dir, archive_file)

        # now delete the source directory
        if delete_source:
            shutil.rmtree(source_dir)

        return archive_file

    def make_archive_on_exit(self,
                             source_dir: str,
                             archive_file: Optional[str] = None,
                             delete_source: bool = True):
        """
        Pack a directory into a zip archive when exiting the experiment context.

        Args:
            source_dir: The relative path of the source directory.
            archive_file: The relative path of the zip archive.
                If not specified, will use `source_dir + ".zip"`.
            delete_source: Whether or not to delete `source_dir` after
                the zip archive has been created?

        See Also:
            :meth:`make_archive()`
        """
        self._require_output_dir()
        self.on_exit.do(lambda: self.make_archive(
            source_dir=source_dir,
            archive_file=archive_file,
            delete_source=delete_source
        ))

    def __enter__(self) -> 'Experiment[TConfig]':
        config_loader = ConfigLoader(self.config)

        # build the argument parser
        if self.args is not None:
            arg_parser = ArgumentParser()
            arg_parser.add_argument(
                '--output-dir', help='Set the experiment output directory.',
                default=NOT_SET, metavar='PATH'
            )
            arg_parser = config_loader.build_arg_parser(arg_parser)
            parsed_args = arg_parser.parse_args(self.args)

            output_dir = parsed_args.output_dir
            if output_dir is not NOT_SET:
                # special hack: override `output_dir` if specified
                self._output_dir = os.path.abspath(output_dir)
                parsed_args.output_dir = NOT_SET

            parsed_args = {
                key: value for key, value in vars(parsed_args).items()
                if value is not NOT_SET
            }
        else:
            parsed_args = {}

        # load previously saved configuration
        if self._output_dir is not None and self._auto_load_config:
            config_files = [
                os.path.join(self.output_dir, 'config.yml'),
                os.path.join(self.output_dir, 'config.json'),
            ]
            for config_file in config_files:
                try:
                    if os.path.exists(config_file):
                        config_loader.load_file(config_file)
                except Exception:  # pragma: no cover
                    raise IOError(f'Failed to load config file: '
                                  f'{config_file!r}')

        # load the cli arguments
        config_loader.load_object(parsed_args)

        # finally, generate the config object
        self._config = config_loader.get(
            discard_undefined=self._discard_undefind_config_fields)

        # prepare for the output dir
        if self._output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        # start the remote doc background worker
        self._remote_doc.start_worker()

        # save the configuration
        if self._auto_save_config:
            self.save_config()

        # trigger the on enter event
        self.on_enter.fire()

        # add to context stack
        _experiment_stack.push(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.on_exit.fire()
        finally:
            if _experiment_stack.top() == self:
                _experiment_stack.pop()
            self._remote_doc.stop_worker()


def get_active_experiment() -> Optional[Experiment]:
    """
    Get the current active experiment object at the top of context stack.

    Returns:
        The active experiment object.
    """
    return _experiment_stack.top()


_experiment_stack: ContextStack[Experiment] = ContextStack()
