import numpy as np


def fft2(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
    return ft / np.sqrt(field.size) if normalize else ft


def ifft2(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ift = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field)))
    return ift * np.sqrt(ift.size) if normalize else ift


def ffrt2(field: np.ndarray, 
          dz: float = 0.0,
          pixel_size: float = 5.04e-6,
          wavelength: float = 1064e-9,
          propagator: np.ndarray = None,
          ) -> np.ndarray:
    if propagator is None:
        propagator = compute_frt_propagator(
            size=field.shape, dz=dz,
            pixel_size=pixel_size, wavelength=wavelength,
            )
    return ifft2(fft2(field) * propagator)


def compute_frt_propagator(size: tuple[int], dz: float, pixel_size: float, wavelength: float) -> np.ndarray:
    _, _, kx, ky = compute_fft_grids(size, pixel_size)
    return np.exp(1j * dz * np.sqrt(np.abs(4 * np.square(np.pi/wavelength) - np.square(kx) - np.square(ky))))


def compute_fft_grids(size: tuple[int], pixel_size: float) -> tuple[np.array]:
    # Spatial plane
    dx, n_pts = pixel_size, size[0]
    grid_size = dx * (n_pts - 1)
    lim_x = n_pts / 2 * dx
    x = np.arange(-lim_x, lim_x, dx)
    x, y = np.meshgrid(x, x)

    # Conjugate plane
    dnx = 1 / grid_size
    lim_nx = (n_pts / 2) * dnx
    kx = 2 * np.pi * np.arange(-lim_nx, lim_nx, dnx)
    kx, ky = np.meshgrid(kx, kx)
    return (x, y, kx, ky)
