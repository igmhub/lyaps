import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

from lyaps.constants import SPEED_LIGHT
from lyaps.deltas import Delta, _available_binnings
from lyaps.utils import userprint

_available_noise_estimators = ["pipeline", "diff"]
_available_resolution_correction = ["mean", "matrix"]


class FourierSpaceDelta(Delta):
    def __init__(
        self,
        wavenumber=None,
        metadata=None,
        delta=None,
        **kwargs,
    ):
        self.wavenumber = wavenumber
        self.wavenumber_weights = np.ones_like(self.wavenumber)
        super(FourierSpaceDelta, self).__init__(
            delta=delta, metadata=metadata, **kwargs
        )

    @classmethod
    def from_real_space_delta(
        cls,
        real_space_delta,
        number_noise_realization=50,
        white_noise_asumption=False,
        rebin_diff_noise=False,
        resolution_correction_method="matrix",
        pixelization_correction=False,
    ):

        delta_arguments = real_space_delta.__dict__.copy()
        delta_arguments.pop("wavelength", None)
        delta_arguments.pop("log_wavelength", None)
        delta_arguments.pop("binning", None)
        delta_arguments.pop("pixel_step", None)

        wavenumber, fourier_delta = transform(
            real_space_delta.delta,
            real_space_delta.binning,
            real_space_delta.pixel_step,
            return_wavenumber=True,
        )

        if delta_arguments.get("diff") is not None:
            exposures_diff = real_space_delta.diff
        else:
            exposures_diff = None

        if rebin_diff_noise and exposures_diff is not None:
            exposures_diff = rebin_diff_noise_function(
                real_space_delta.wavelength,
                exposures_diff,
                real_space_delta.pixel_step,
            )

        fourier_delta_noise, power_spectrum_noise, fourier_delta_diff = (
            compute_noise_power_spectra(
                real_space_delta.wavelength,
                real_space_delta.ivar,
                real_space_delta.binning,
                real_space_delta.pixel_step,
                exposures_diff=exposures_diff,
                number_noise_realization=number_noise_realization,
                white_noise_asumption=white_noise_asumption,
            )
        )

        resolution_correction = compute_resolution_correction(
            resolution_correction_method,
            wavenumber,
            real_space_delta.pixel_step,
            real_space_delta.binning,
            resolution_matrix=real_space_delta.resomat,
            mean_resolution=real_space_delta.metadata["MEANRESO"],
            pixelization_correction=pixelization_correction,
        )

        if "weight" in delta_arguments:
            fourier_weight = real_space_delta.weight
            _, fourier_weight = transform(
                real_space_delta.weight,
                real_space_delta.binning,
                real_space_delta.pixel_step,
                return_wavenumber=False,
            )
            power_spectrum_weigth = fourier_weight.real**2 + fourier_weight.imag**2
        else:
            fourier_weight = None
            power_spectrum_weigth = None

        additional_fields = {
            "delta_noise": fourier_delta_noise,
            "power_spectrum_noise": power_spectrum_noise,
            "delta_diff": fourier_delta_diff,
            "resolution_correction": resolution_correction,
            "delta_weight": fourier_weight,
            "power_spectrum_weight": power_spectrum_weigth,
        }
        return cls(
            wavenumber=wavenumber,
            delta=fourier_delta,
            **additional_fields,
        )

    def plot_power_spectrum(
        self,
        ax=None,
    ):

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.wavenumber, self.power_spectrum)
        legend = ["Power Spectrum"]
        if hasattr(self, "power_spectrum_weight"):
            ax.plot(self.wavenumber, self.power_spectrum_weight)
            legend.append("Weight")
        ax.legend(legend)
        ax.set_xlabel("Wavenumer")
        return ax

    @property
    def power_spectrum_raw(self):
        return self.delta.real**2 + self.delta.imag**2

    @property
    def power_spectrum_diff(self):
        if "delta_diff" in self.__dict__:
            return self.delta_diff.real**2 + self.delta_diff.imag**2
        else:
            return None

    @property
    def power_spectrum(self, noise_estimator="pipeline"):
        if noise_estimator == "pipeline":
            return (
                self.power_spectrum_raw - self.power_spectrum_noise
            ) / self.resolution_correction
        elif noise_estimator == "diff":
            if "delta_diff" in self.__dict__:
                return (
                    self.power_spectrum_raw - self.power_spectrum_diff
                ) / self.resolution_correction
            else:
                raise ValueError(
                    "The delta_diff is not available, cannot compute power_spectrum with diff noise estimator"
                )
        else:
            raise ValueError(
                f"Unknown noise estimator {noise_estimator}, available are {_available_noise_estimators}"
            )


def transform(
    delta_array,
    binning,
    pixel_step,
    return_wavenumber=True,
):
    number_pixels = len(delta_array)
    if binning == "linear":
        length_lambda = pixel_step * number_pixels
    elif binning == "log10":
        length_lambda = pixel_step * SPEED_LIGHT * np.log(10.0) * number_pixels
    else:
        raise ValueError(f"Binning must be {_available_binnings}, got {binning}")

    num_pixels = number_pixels
    fourier_delta = rfft(delta_array)
    fourier_delta = fourier_delta * np.sqrt(length_lambda) / num_pixels
    if return_wavenumber:
        wavenumber = (
            2
            * np.pi
            * rfftfreq(
                num_pixels,
                length_lambda / num_pixels,
            )
        )
    else:
        wavenumber = None

    return wavenumber, fourier_delta


def compute_noise_power_spectra(
    wavelength,
    inversed_variance,
    binning,
    pixel_step,
    exposures_diff=None,
    number_noise_realization=500,
    white_noise_asumption=False,
):

    num_pixels = len(wavelength)
    num_bins_fft = num_pixels // 2 + 1

    average_delta_noise = np.zeros(num_bins_fft, dtype=np.complex128)
    average_power_spectrum_noise = np.zeros(num_bins_fft)
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
        average_delta_noise += fourier_delta_noise.real**2
        average_power_spectrum_noise += (
            fourier_delta_noise.real**2 + fourier_delta_noise.imag**2
        )

    average_delta_noise /= float(number_noise_realization)
    average_power_spectrum_noise /= float(number_noise_realization)
    if white_noise_asumption:
        white_noise_value = np.mean(average_power_spectrum_noise)
        average_power_spectrum_noise = np.full_like(
            average_power_spectrum_noise, white_noise_value
        )
    if exposures_diff is not None:
        _, fourier_delta_diff = transform(
            exposures_diff,
            binning,
            pixel_step,
            return_wavenumber=False,
        )
    else:
        fourier_delta_diff = None

    return average_delta_noise, average_power_spectrum_noise, fourier_delta_diff


def rebin_diff_noise_function(
    wavelength,
    exposures_diff,
    pixel_step,
):

    rebin = 3
    if exposures_diff.size < rebin:
        userprint("Warning: exposures_diff.size too small for rebin")
        return exposures_diff
    rebin_delta_lambda_or_log_lambda = rebin * pixel_step

    bins = np.floor(
        (wavelength - wavelength.min()) / rebin_delta_lambda_or_log_lambda + 0.5
    ).astype(int)

    rebin_exposure_diff = np.bincount(bins.astype(int), weights=exposures_diff)
    rebin_counts = np.bincount(bins.astype(int))
    w = rebin_counts > 0
    if len(rebin_counts) == 0:
        userprint("Error: exposures_diff size = 0 ", exposures_diff)
    rebin_exposure_diff = rebin_exposure_diff[w] / np.sqrt(rebin_counts[w])

    rebin_exposure_diff_out = np.zeros(exposures_diff.size)
    for index in range(len(exposures_diff) // len(rebin_exposure_diff) + 1):
        length_max = min(len(exposures_diff), (index + 1) * len(rebin_exposure_diff))
        rebin_exposure_diff_out[index * len(rebin_exposure_diff) : length_max] = (
            rebin_exposure_diff[: (length_max - index * len(rebin_exposure_diff))]
        )
        np.random.shuffle(rebin_exposure_diff)

    return rebin_exposure_diff_out


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

    resolution_matrix = np.atleast_2d(resolution_matrix)

    w2_arr = []
    for resmat in resolution_matrix:
        if np.any(resmat) != 0.0:
            r = np.append(resmat, np.zeros(resolution_matrix.shape[0] - resmat.size))
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
            f"The available resolution correction methods are {_available_resolution_correction}",
            f"{correction_method} was given",
        )
    return resolution_correction


def fourier_transform_one_delta_object(
    delta_object,
    fourier_config,
):

    number_noise_realization = fourier_config.getint("number noise realization")
    white_noise_asumption = fourier_config.getboolean("white noise assumption")
    rebin_diff_noise = fourier_config.getboolean("rebin diff noise")
    resolution_correction_method = fourier_config.get("resolution correction method")
    pixelization_correction = fourier_config.getboolean("pixelization correction")

    return FourierSpaceDelta.from_real_space_delta(
        delta_object,
        number_noise_realization=number_noise_realization,
        white_noise_asumption=white_noise_asumption,
        rebin_diff_noise=rebin_diff_noise,
        resolution_correction_method=resolution_correction_method,
        pixelization_correction=pixelization_correction,
    )
