import numpy as np
import torch
import tensorflow as tf


def pearson(x, y, squared: bool = False, inversed: bool = False) -> torch.float:
    x, y = torch.abs(x), torch.abs(y)
    if squared:
            x, y = torch.square(x), torch.square(y)
    s = torch.sum((x - torch.mean(x)) * (y - torch.mean(y)) / torch.tensor(x.numel()))
    p = s / (torch.std(x) * torch.std(y))
    return 1 - p if inversed else p


def quality(x, y, inversed: bool = True, squared: bool = True):
        prod = torch.sum(x * torch.conj(y))
        norm = torch.sum(torch.abs(x) * torch.abs(y))
        q = torch.abs(prod / norm)

        if squared:
            q = torch.square(q)
        if inversed:
            q = 1 - q
        return q


def mae_numpy(x: np.ndarray, y: np.ndarray):
    """Return the mean absolute error between the two arrays."""
    if np.iscomplexobj(x):
        x = np.abs(x)
    if np.iscomplexobj(y):
        y = np.abs(y)
    return np.mean(np.abs(x - y))


def mse_numpy(x: np.ndarray, y: np.ndarray):
    """Return the mean square error between the two arrays."""
    if np.iscomplexobj(x):
        x = np.abs(x)
    if np.iscomplexobj(y):
        y = np.abs(y)
    return np.mean(np.square(x - y))


def dot_product_numpy(x: np.ndarray, y: np.ndarray, normalized: bool = True):
    """Return the scalar product between the two complex arrays."""
    prod = np.sum(x * np.conjugate(y))
    norm = np.sum(np.abs(x) * np.abs(y))
    return prod / norm if normalized else prod
