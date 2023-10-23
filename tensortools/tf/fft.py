import numpy as np
import tensorflow as tf

pi = np.pi


def fft2(field: tf.Tensor, normalize: bool = True) -> tf.Tensor:
    ft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(field)))
    return ft / tf.cast(tf.math.sqrt(tf.cast(tf.size(ft), tf.float64)), ft.dtype) if normalize else ft


def ifft2(field: tf.Tensor, normalize: bool = True) -> tf.Tensor:
    ift = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(field)))
    return ift * tf.cast(tf.math.sqrt(tf.cast(tf.size(ift), tf.float64)), ift.dtype) if normalize else ift


def ffrt2(field: tf.Tensor, 
          dz: float = 0.0,
          pixel_size: float = 5.04e-6,
          wavelength: float = 1064e-9,
          propagator: tf.Tensor = None,
          ) -> tf.Tensor:
    if propagator is None:
        propagator = compute_frt_propagator(
            size=field.shape, dz=dz,
            pixel_size=pixel_size, wavelength=wavelength,
            )
    return ifft2(fft2(field) * propagator)


def compute_frt_propagator(size: tuple[int], dz: float, pixel_size: float, wavelength: float) -> tf.Tensor:
    _, _, kx, ky = compute_fft_grids(size, pixel_size)
    return tf.math.exp(1j * tf.cast(dz * tf.math.sqrt(tf.math.abs(4 * tf.cast(tf.math.square(pi/wavelength), kx.dtype) - tf.math.square(kx) - tf.math.square(ky))), tf.complex128))


def compute_fft_grids(size: tuple[int], pixel_size: float) -> tuple[tf.Tensor]:
    # Spatial plane
    dx, n_pts = pixel_size, size[0]
    grid_size = dx * (n_pts - 1)
    lim_x = n_pts / 2 * dx
    x = tf.experimental.numpy.arange(-lim_x, lim_x, dx)
    x, y = tf.experimental.numpy.meshgrid(x, x)

    # Conjugate plane
    dnx = 1 / grid_size
    lim_nx = (n_pts / 2) * dnx
    kx = 2 * pi * tf.experimental.numpy.arange(-lim_nx, lim_nx, dnx)
    kx, ky = tf.experimental.numpy.meshgrid(kx, kx)
    return (x, y, kx, ky)
