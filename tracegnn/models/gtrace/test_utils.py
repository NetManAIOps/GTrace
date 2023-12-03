from urllib.error import HTTPError
from typing import *

import mltk
import torch
import yaml
from tracegnn.models.gtrace.dataset import init_config

from tracegnn.models.gtrace.models.level_model import LevelModel
from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.utils import *
from loguru import logger


__all__ = [
    'load_config',
    'load_model',
    'load_model2',
]


def _model_and_config_file(model_path: str) -> Tuple[str, str]:
    # get model file and config file path
    if model_path.endswith('.pt'):
        model_file = model_path
        config_file = model_path.rsplit('/', 2)[-3] + '/config.json'
    else:
        if not model_path.endswith('/'):
            model_path += '/'
        model_file = model_path + 'model.pth'
        config_file = model_path + 'config.json'

    return model_file, config_file


def load_config(model_path: str, strict: bool, extra_args) -> ExpConfig:
    # get model file and config file path
    model_file, config_file = _model_and_config_file(model_path)

    # load config
    with as_local_file(config_file) as config_file:
        config_loader = mltk.ConfigLoader(ExpConfig)
        config_loader.load_file(config_file)

    # also patch the config
    if extra_args:
        extra_args_dict = {}
        for arg in extra_args:
            if arg.startswith('--'):
                arg = arg[2:]
                if '=' not in arg:
                    val = True
                else:
                    arg, val = arg.split('=', 1)
                    val = yaml.safe_load(val)
                extra_args_dict[arg] = val
            else:
                raise ValueError(f'Unsupported argument: {arg!r}')
        config_loader.load_object(extra_args_dict)

    # get the config
    if strict:
        discard_undefined = mltk.type_check.DiscardMode.NO
    else:
        discard_undefined = mltk.type_check.DiscardMode.WARN
    return config_loader.get(discard_undefined=discard_undefined)


def load_model(model_path: str,
               strict: bool=True,
               device: str=None,
               extra_args=None,
               n_z=5
               ) -> Tuple[LevelModel, ExpConfig]:
    # load config
    train_config = load_config(model_path, strict, extra_args)
    train_config.Model.n_z = n_z

    if device is not None:
        train_config.device = device

    init_config(train_config)

    # load model
    model = load_model2(model_path, train_config)
    return model, train_config


def load_model2(model_path: str,
                train_config: ExpConfig) -> LevelModel:
    # get model file and config file path
    model_file, config_file = _model_and_config_file(model_path)
    print(model_file)
    # load the model
    model = LevelModel(train_config)
    try:
        with as_local_file(model_file.replace('model.pth', 'es.pt')) as model_file:
            model.load_state_dict(torch.load(
                model_file,
                map_location=train_config.device
            ))
    except HTTPError as ex:
        if ex.code != 404:
            raise
        with as_local_file(model_file) as model_file:
            model.load_state_dict(torch.load(
                model_file,
                map_location='cpu'
            ))
            model.to(train_config.device)
            logger.info(f'Model loaded to {train_config.device}.')

    return model
