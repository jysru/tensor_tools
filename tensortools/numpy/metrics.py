import numpy as np


def pearson(x: np.ndarray, y: np.ndarray, squared: bool = False, inversed: bool = False) -> float:
    x, y = np.abs(x), np.abs(y)
    if squared:
        x, y = np.square(x), np.square(y)
    s = np.sum((x - np.mean(x)) * (y - np.mean(y)) / x.size)
    p = s / (np.std(x) * np.std(y))
    return 1 - p if inversed else p


def mae(x: np.ndarray, y: np.ndarray) -> float:
    """Return the mean absolute error between the two arrays."""
    if np.iscomplexobj(x):
        x = np.abs(x)
    if np.iscomplexobj(y):
        y = np.abs(y)
    return np.mean(np.abs(x - y))


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Return the mean square error between the two arrays."""
    if np.iscomplexobj(x):
        x = np.abs(x)
    if np.iscomplexobj(y):
        y = np.abs(y)
    return np.mean(np.square(x - y))


def dot_product(x: np.ndarray, y: np.ndarray, normalized: bool = True) -> float:
    """Return the scalar product between the two complex arrays."""
    prod = np.sum(x * np.conjugate(y))
    norm = np.sum(np.abs(x) * np.abs(y))
    return prod / norm if normalized else prod


def quality(x: np.ndarray, y: np.ndarray, squared: bool = False, inversed: bool = False) -> float:
    """Return the magnitude of the normalized dot product between the two complex arrays."""
    q = np.abs(dot_product(x, y, normalized=True))
    if squared:
        q = np.square(q)
    return 1 - q if inversed else q

