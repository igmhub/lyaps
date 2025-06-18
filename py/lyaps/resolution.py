import numpy as np

from lyaps.fourier import transform


def compute_resolution_correction_mean(
    wavenumber,
    pixel_step,
    mean_resolution,
    pixelization_correction=False,
):
    num_bins_fourier = len(wavenumber)
    resolution_correction = np.ones(num_bins_fourier)
    resolution_correction *= np.exp(-((wavenumber * mean_resolution) ** 2))

    if pixelization_correction:
        pixelization_factor = np.sinc(wavenumber * pixel_step / (2 * np.pi)) ** 2
        resolution_correction *= pixelization_factor
    return resolution_correction


def compute_resolution_correction_matrix(
    wavenumber,
    resolution_matrix,
    pixel_step,
    binning,
    pixelization_correction=False,
):

    # this allows either computing the power for each pixel seperately or for the mean
    resolution_matrix = np.atleast_2d(resolution_matrix)

    w2_arr = []
    # first compute the power in the resmat for each pixel, then average
    for resmat in resolution_matrix:
        r = np.append(resmat)
        _, fourier_resmat = transform(
            r,
            binning,
            pixel_step,
            return_wavenumber=False,
        )
        w2 = fourier_resmat.real**2 + fourier_resmat.imag**2
        w2 /= w2[0]
        w2_arr.append(w2)

    w_res2 = np.mean(w2_arr, axis=0)

    # the following assumes that the resolution matrix is storing the actual
    # resolution convolved with the pixelization kernel along each matrix axis
    resolution_correction = w_res2
    if pixelization_correction:
        pixelization_factor = np.sinc(wavenumber * pixel_step / (2 * np.pi)) ** 2
        resolution_correction /= pixelization_factor

    return resolution_correction


def compute_resolution_correction(
    correction_method,
    wavenumber,
    pixel_step,
    binning,
    resolution_matrix=None,
    mean_resolution=None,
    pixelization_correction=False,
):
    if correction_method == "mean":
        if mean_resolution is None:
            raise ValueError(
                "The method mean for resolution correction needs mean_resolution"
            )
        else:
            resolution_correction = compute_resolution_correction_mean(
                wavenumber,
                pixel_step,
                mean_resolution,
                pixelization_correction=pixelization_correction,
            )
    elif correction_method == "matrix":
        if resolution_matrix is None:
            raise ValueError(
                "The method matrix for resolution correction needs resolution_matrix"
            )
        resolution_correction = compute_resolution_correction_matrix(
            wavenumber,
            resolution_matrix,
            pixel_step,
            binning,
            pixelization_correction=False,
        )
    else:
        raise ValueError(
            "The available resolution correction methods are {_available_resolution_correction}",
            f"{correction_method} was given",
        )
    return resolution_correction
