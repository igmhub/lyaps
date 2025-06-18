import numpy as np

from py.lyaps.fourier_old import transform


def compute_correction_reso(
    delta_pixel,
    mean_reso,
    k,
):
    num_bins_fft = len(k)
    correction = np.ones(num_bins_fft)

    pixelization_factor = np.sinc(k * delta_pixel / (2 * np.pi)) ** 2

    correction *= np.exp(-((k * mean_reso) ** 2))
    correction *= pixelization_factor
    return correction


def compute_correction_reso_matrix(
    reso_matrix,
    binning,
    k,
    pixel_step,
    num_pixel,
    pixelization_correction=False,
):

    # this allows either computing the power for each pixel seperately or for the mean
    reso_matrix = np.atleast_2d(reso_matrix)

    w2_arr = []
    # first compute the power in the resmat for each pixel, then average
    for resmat in reso_matrix:
        r = np.append(resmat, np.zeros(num_pixel - resmat.size))
        k_resmat, fourier_resmat = transform(
            r,
            binning,
            pixel_step,
            return_wavenumber=True,
        )
        w2 = fourier_resmat.real**2 + fourier_resmat.imag**2
        try:
            assert np.all(k_resmat == k)
        except AssertionError as error:
            raise AssertionError(
                "for some reason the resolution matrix correction has "
                "different k scaling than the pk"
            ) from error
        w2 /= w2[0]
        w2_arr.append(w2)

    w_res2 = np.mean(w2_arr, axis=0)

    # the following assumes that the resolution matrix is storing the actual
    # resolution convolved with the pixelization kernel along each matrix axis
    correction = w_res2
    if pixelization_correction:
        pixelization_factor = np.sinc(k * pixel_step / (2 * np.pi)) ** 2
        correction /= pixelization_factor

    return correction
