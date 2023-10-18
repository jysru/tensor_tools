import numpy as np
import torch
import tensorflow as tf


def pearson(y_true: tf.Tensor, y_pred: tf.Tensor, squared: bool = False, inversed: bool = False) -> tf.Tensor:
    y_true, y_pred = tf.math.abs(y_true), tf.math.abs(y_pred)
    if squared:
        y_true, y_pred = tf.math.square(y_true), tf.math.square(y_pred)
    s = tf.math.reduce_sum((y_true - tf.math.reduce_mean(y_true)) * (y_pred - tf.math.reduce_mean(y_pred)) / tf.cast(tf.size(y_true), tf.float32))
    p = s / (tf.math.reduce_std(y_true) * tf.math.reduce_std(y_pred))
    return 1 - p if inversed else p


def quality_torch(x, y, inversed: bool = True, squared: bool = True):
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


def quality_numpy(x: np.ndarray, y: np.ndarray, squared: bool = True, inversed: bool = False):
    """Return the magnitude of the normalized dot product between the two complex arrays."""
    q = np.abs(dot_product_numpy(x, y, normalized=True))
    if squared:
        q = np.square(q)
    if inversed:
        q = 1 - q
    return q


