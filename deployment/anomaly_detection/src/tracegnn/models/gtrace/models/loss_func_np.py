import numpy as np
import numba as nb
from numba import types
from tracegnn.models.gtrace.config import ExpConfig


@nb.njit(fastmath=True, parallel=True)
def normal_loss_np(label: np.ndarray,
                mu: np.ndarray,
                logvar: np.ndarray,
                eps: float=1e-7) -> np.ndarray:
    """
        Calculate the loss with normal distribution.
    """

    loss = (mu - label) ** 2 / (2 * np.exp(logvar) + eps) + 0.5 * logvar

    return loss[..., 0]


@nb.njit(types.float64[:](types.float64[:, :]), fastmath=True, parallel=True)
def log_exp_mean_np(x: np.ndarray) -> np.ndarray:
    x_min = np.zeros((x.shape[0], 1), dtype=np.float64)

    for i in range(x.shape[0]):
        x_min[i, 0] = np.min(x[i])

    y = np.clip(x - x_min, -np.inf, 20.0)
    result = np.log(np.sum(np.exp(y), axis=1) / y.shape[1] + 1e-7)

    return result + x_min[:, 0]
