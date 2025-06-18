import numpy as np

from lyaps.fourier import transform
from lyaps.utils import userprint


def compute_noise_power_spectra(
    wavelength,
    inversed_variance,
    binning,
    pixel_step,
    exposures_diff=None,
    number_noise_realization=50,
):

    num_pixels = len(wavelength)
    num_bins_fft = num_pixels // 2 + 1

    average_delta_noise = np.zeros(num_bins_fft)
    pipeline_error = np.zeros(num_pixels)
    mask = inversed_variance > 0
    pipeline_error[mask] = 1.0 / np.sqrt(inversed_variance[mask])

    for _ in range(number_noise_realization):
        delta_noise = np.zeros(num_pixels)
        delta_noise[mask] = np.random.normal(0.0, pipeline_error[mask])
        _, fourier_delta_noise = transform(
            delta_noise,
            binning,
            pixel_step,
            return_wavenumber=False,
        )

        average_delta_noise += fourier_delta_noise
        average_delta_noise /= float(number_noise_realization)

    if exposures_diff is not None:
        _, fourier_delta_diff = transform(
            exposures_diff,
            binning,
            pixel_step,
            return_wavenumber=False,
        )
    else:
        fourier_delta_diff = None

    return average_delta_noise, fourier_delta_diff


def rebin_diff_noise(
    wavelength,
    exposures_diff,
    pixel_step,
):

    rebin = 3
    if exposures_diff.size < rebin:
        userprint("Warning: exposures_diff.size too small for rebin")
        return exposures_diff
    rebin_delta_lambda_or_log_lambda = rebin * pixel_step

    # rebin not mixing pixels separated by masks
    bins = np.floor(
        (wavelength - wavelength.min()) / rebin_delta_lambda_or_log_lambda + 0.5
    ).astype(int)

    rebin_exposure_diff = np.bincount(bins.astype(int), weights=exposures_diff)
    rebin_counts = np.bincount(bins.astype(int))
    w = rebin_counts > 0
    if len(rebin_counts) == 0:
        userprint("Error: exposures_diff size = 0 ", exposures_diff)
    rebin_exposure_diff = rebin_exposure_diff[w] / np.sqrt(rebin_counts[w])

    # now merge the rebinned array into a noise array
    rebin_exposure_diff_out = np.zeros(exposures_diff.size)
    for index in range(len(exposures_diff) // len(rebin_exposure_diff) + 1):
        length_max = min(len(exposures_diff), (index + 1) * len(rebin_exposure_diff))
        rebin_exposure_diff_out[index * len(rebin_exposure_diff) : length_max] = (
            rebin_exposure_diff[: (length_max - index * len(rebin_exposure_diff))]
        )
        # shuffle the array before the next iteration
        np.random.shuffle(rebin_exposure_diff)

    return rebin_exposure_diff_out
