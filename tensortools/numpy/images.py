import numpy as np



def pooling_2d(feature_map: np.ndarray, kernel: tuple = (2, 2), func: callable = np.max) -> np.ndarray:
    """
    Applies 2D pooling to a feature map.

    Parameters
    ----------
    feature_map : np.ndarray
        A 2D or 3D feature map to apply max pooling to. If the feature map is 3D, the channels should be the first dimension.
    kernel: tuple
        The size of the kernel to use for max pooling.
    func:
        Numpy reduction method to apply: default = numpy.max

    Returns
    -------
    np.ndarray
        The feature map after pooling was applied.
    """

    dim_add = 1 if feature_map.ndim > 2 else 0

    # Check if it fits without padding the feature map
    if feature_map.shape[0 + dim_add] % kernel[0] != 0:
        # Add padding to the feature map
        feature_map = np.pad(feature_map, ((0, kernel[0] - feature_map.shape[0 + dim_add] % kernel[0]), (0,0), (0,0)), 'constant')
    
    if feature_map.shape[1 + dim_add] % kernel[1] != 0:
        feature_map = np.pad(feature_map, ((0, 0), (0, kernel[1] - feature_map.shape[1 + dim_add] % kernel[1]), (0,0)), 'constant')

    if dim_add:
        newshape = (-1, feature_map.shape[1] // kernel[0], kernel[0], feature_map.shape[2] // kernel[1], kernel[1])
    else:
        newshape = (feature_map.shape[0] // kernel[0], kernel[0], feature_map.shape[1] // kernel[1], kernel[1])

    pooled = feature_map.reshape(newshape)
    pooled = func(pooled, axis=(1 + dim_add, 3 + dim_add))
    return pooled


def max_pooling_2d(feature_map: np.ndarray, kernel: tuple = (2, 2)) -> np.ndarray:
    return pooling_2d(feature_map, kernel, func=np.max)


def avg_pooling_2d(feature_map: np.ndarray, kernel: tuple = (2, 2)) -> np.ndarray:
    return pooling_2d(feature_map, kernel, func=np.mean)
