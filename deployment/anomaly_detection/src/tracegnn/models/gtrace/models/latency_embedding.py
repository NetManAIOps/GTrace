import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from ..config import ExpConfig
from typing import *


@torch.jit.script
def class_latency_to_feature(latency: torch.Tensor, 
                       length: int,
                       max_value: float) -> torch.Tensor:
    """
        latency: [b]
        return: [b x length]

        The last element is for > max_value.
    """
    interval = max_value / (length - 1)
    result = torch.minimum((latency / interval).long(),
        torch.tensor(length - 1, dtype=torch.long, device=latency.device))

    return result

@torch.jit.script
def class_feature_to_latency(feature: torch.Tensor,
                       max_value: float) -> torch.Tensor:
    
    """
        feature: [b x length]
        return: [b]
    """
    interval = max_value / (feature.size(-1) - 1)
    feature = torch.softmax(feature, dim=-1)
    time_steps = torch.arange(0, feature.size(-1), dtype=torch.float, device=feature.device)

    result = (feature @ time_steps.unsqueeze(1)).squeeze(1)
    result = (result + 0.5) * interval

    return result


@torch.jit.script
def log_latency_to_feature(latency: torch.Tensor, 
                       length: int,
                       max_value: float) -> torch.Tensor:
    """
        latency: [b]
        return: [b x length]
    """
    result = torch.log(latency + 1.0) / torch.log(torch.tensor(10.0))

    # normalize
    result = result / torch.log10(torch.tensor(max_value)) * 2 - 1.0

    return result.unsqueeze(1)

@torch.jit.script
def log_feature_to_latency(feature: torch.Tensor,
                       max_value: float) -> torch.Tensor:
    
    """
        feature: [b x length]
        return: [b]
    """
    # normalize
    feature = (feature + 1.0) / 2.0 * torch.log10(torch.tensor(max_value))

    result = 10.0 ** feature[..., 0] - 1.0
    return result


@torch.jit.script
def simple_latency_to_feature(latency: torch.Tensor, 
                       length: int,
                       max_value: float,
                       clip: bool=False) -> torch.Tensor:
    """
        latency: [b]
        return: [b x length]
    """
    # Set the maximum value to 5 to avoid nan
    if clip:
        result = torch.minimum(latency / max_value,
                torch.ones_like(latency) * max_value * 5)
    else:
        result = latency / max_value

    return result.unsqueeze(1)

@torch.jit.script
def simple_feature_to_latency(feature: torch.Tensor,
                       max_value: float) -> torch.Tensor:
    
    """
        feature: [b x length]
        return: [b]
    """
    # normalize
    result = feature * max_value

    return result[..., 0]


@torch.jit.script
def normal_latency_to_feature(latency: torch.Tensor, 
                       length: int,
                       latency_range: torch.Tensor,
                       operation_id: torch.Tensor,
                       clip: bool=False) -> torch.Tensor:
    """
        latency: [b]
        return: [b x length]
    """
    mean = latency_range[operation_id, 0]
    std = latency_range[operation_id, 1]

    result = (latency - mean) / std

    if clip: 
        result = torch.minimum(result, torch.ones_like(result) * 5)

    return result.unsqueeze(1)

@torch.jit.script
def normal_feature_to_latency(feature: torch.Tensor,
                        latency_range: torch.Tensor,
                        operation_id: torch.Tensor) -> torch.Tensor:
    
    """
        feature: [b x length]
        return: [b]
    """
    mean = latency_range[operation_id, 0]
    std = latency_range[operation_id, 1]

    result = feature * std + mean

    return result[..., 0]


@torch.jit.script
def vector_latency_to_feature(latency: torch.Tensor, 
                       length: int,
                       max_value: float) -> torch.Tensor:
    """
        latency: [b]
        return: [b x length]
    """
    result = torch.empty([latency.size(0), length], dtype=torch.float, device=latency.device)
    interval = max_value ** (1.0 / length)
    zeros = torch.tensor([0.0], device=latency.device)

    cur_max_value = interval
    remain_latency = latency.clone()

    for i in range(length-1):
        result[:, i] = remain_latency % cur_max_value / cur_max_value * 2.0 - 1.0
        remain_latency = torch.maximum(zeros, remain_latency - remain_latency % cur_max_value)
        cur_max_value *= interval

    result[:, length-1] = remain_latency / cur_max_value * 2.0 - 1.0

    return result

@torch.jit.script
def vector_feature_to_latency(feature: torch.Tensor,
                       max_value: float) -> torch.Tensor:
    
    """
        feature: [b x length]
        return: [b]
    """
    result = torch.zeros([feature.size(0)], dtype=torch.float, device=feature.device)
    interval = max_value ** (1.0 / feature.size(1))

    cur_max_value = interval
    for i in range(feature.size(1)):
        result += (feature[:, i] / 2 + 0.5) * cur_max_value
        cur_max_value *= interval

    return result


def latency_to_feature(config: ExpConfig,
                       latency: torch.Tensor,
                       operation_id: torch.Tensor=None,
                       clip: bool=False) -> torch.Tensor:
    if config.Latency.embedding_type == 'class':
        return class_latency_to_feature(latency, config.Latency.latency_feature_length, config.Latency.latency_max_value)
    elif config.Latency.embedding_type == 'vector':
        return vector_latency_to_feature(latency, config.Latency.latency_feature_length, config.Latency.latency_max_value)
    elif config.Latency.embedding_type == 'log':
        return log_latency_to_feature(latency, config.Latency.latency_feature_length, config.Latency.latency_max_value)
    elif config.Latency.embedding_type == 'normal':
        return normal_latency_to_feature(latency, config.Latency.latency_feature_length, 
            config.RuntimeInfo.latency_range, operation_id, clip)
    else:
        return simple_latency_to_feature(latency, config.Latency.latency_feature_length, config.Latency.latency_max_value)


def feature_to_latency(config: ExpConfig,
                       feature: torch.Tensor,
                       operation_id: torch.Tensor=None) -> torch.Tensor:
    if config.Latency.embedding_type == 'class':
        return class_feature_to_latency(feature, config.Latency.latency_max_value)
    elif config.Latency.embedding_type == 'vector':
        return vector_feature_to_latency(feature, config.Latency.latency_max_value)
    elif config.Latency.embedding_type == 'log':
        return log_feature_to_latency(feature, config.Latency.latency_max_value)
    elif config.Latency.embedding_type == 'normal':
        return normal_feature_to_latency(feature, config.RuntimeInfo.latency_range, operation_id)
    else:
        return simple_feature_to_latency(feature, config.Latency.latency_max_value)
