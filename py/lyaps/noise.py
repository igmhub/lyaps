from lyaps.utils import userprint


def compute_pk_noise(
    delta_lambda_or_log_lambda,
    ivar,
    exposures_diff,
    run_noise,
    num_noise_exposures=10,
    linear_binning=False,
):
    """Compute the noise power spectrum

    Two noise power spectrum are computed: one using the pipeline noise and
    another one using the noise derived from exposures_diff

    Arguments
    ---------
    delta_lambda_or_log_lambda: float
    Variation of the logarithm of the wavelength between two pixels

    ivar: array of float
    Array containing the inverse variance

    exposures_diff: array of float
    Semidifference between two customized coadded spectra obtained from
    weighted averages of the even-number exposures, for the first
    spectrum, and of the odd-number exposures, for the second one

    run_noise: boolean
    If False the noise power spectrum using the pipeline noise is not
    computed and an array filled with zeros is returned instead

    num_noise_exposures: int
    Number of exposures to average for noise power estimate

    Return
    ------
    pk_noise: array of float
    The noise Power Spectrum using the pipeline noise

    pk_diff: array of float
    The noise Power Spectrum using the noise derived from exposures_diff

    fft_delta_noise: array of float
    The Fourier transform of noise pipeline realization

    fft_delta_diff: array of float
    The Fourier transform of the exposure difference delta
    """
    num_pixels = len(ivar)
    num_bins_fft = num_pixels // 2 + 1

    pk_noise = np.zeros(num_bins_fft)
    fft_delta_noise = np.zeros(num_bins_fft)
    error = np.zeros(num_pixels)
    w = ivar > 0
    error[w] = 1.0 / np.sqrt(ivar[w])

    if run_noise:
        for _ in range(num_noise_exposures):
            delta_exp = np.zeros(num_pixels)
            delta_exp[w] = np.random.normal(0.0, error[w])
            _, fft_delta_noise, pk_exp = compute_pk_raw(
                delta_lambda_or_log_lambda, delta_exp, linear_binning=linear_binning
            )
            pk_noise += pk_exp

        pk_noise /= float(num_noise_exposures)

    _, fft_delta_diff, pk_diff = compute_pk_raw(
        delta_lambda_or_log_lambda, exposures_diff, linear_binning=linear_binning
    )

    return pk_noise, pk_diff, fft_delta_noise, fft_delta_diff


def rebin_diff_noise(pixel_step, lambda_or_log_lambda, exposures_diff):
    """Rebin the semidifference between two customized coadded spectra to
    construct the noise array

    Note that inputs can be either linear or log-lambda spaced units (but
    pixel_step and lambda_or_log_lambda need the same unit)

    The rebinning is done by combining 3 of the original pixels into analysis
    pixels.

    Arguments
    ---------
    pixel_step: float
    Variation of the logarithm of the wavelength between two pixels
    for linear binnings this would need to be the wavelength difference

    lambda_or_log_lambda: array of float
    Array containing the logarithm of the wavelengths (in Angs)
    for linear binnings this would need to be just wavelength

    exposures_diff: array of float
    Semidifference between two customized coadded spectra obtained from
    weighted averages of the even-number exposures, for the first
    spectrum, and of the odd-number exposures, for the second one

    Return
    ------
    noise: array of float
    The noise array
    """
    rebin = 3
    if exposures_diff.size < rebin:
        userprint("Warning: exposures_diff.size too small for rebin")
        return exposures_diff
    rebin_delta_lambda_or_log_lambda = rebin * pixel_step

    # rebin not mixing pixels separated by masks
    bins = np.floor(
        (lambda_or_log_lambda - lambda_or_log_lambda.min())
        / rebin_delta_lambda_or_log_lambda
        + 0.5
    ).astype(int)

    rebin_exposure_diff = np.bincount(bins.astype(int), weights=exposures_diff)
    rebin_counts = np.bincount(bins.astype(int))
    w = rebin_counts > 0
    if len(rebin_counts) == 0:
        userprint("Error: exposures_diff size = 0 ", exposures_diff)
    rebin_exposure_diff = rebin_exposure_diff[w] / np.sqrt(rebin_counts[w])

    # now merge the rebinned array into a noise array
    noise = np.zeros(exposures_diff.size)
    for index in range(len(exposures_diff) // len(rebin_exposure_diff) + 1):
        length_max = min(len(exposures_diff), (index + 1) * len(rebin_exposure_diff))
        noise[index * len(rebin_exposure_diff) : length_max] = rebin_exposure_diff[
            : (length_max - index * len(rebin_exposure_diff))
        ]
        # shuffle the array before the next iteration
        np.random.shuffle(rebin_exposure_diff)

    return noise
