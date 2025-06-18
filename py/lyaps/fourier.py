import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

from lyaps.constants import SPEED_LIGHT
from lyaps.deltas import Delta, _available_binnings
from lyaps.noise import compute_noise_power_spectra
from lyaps.resolution import compute_resolution_correction


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
        resolution_correction_method="matrix",
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

        fourier_delta_noise, fourier_delta_diff = compute_noise_power_spectra(
            real_space_delta.wavelength,
            real_space_delta.ivar,
            real_space_delta.binning,
            real_space_delta.pixel_step,
            exposures_diff=exposures_diff,
            number_noise_realization=number_noise_realization,
        )

        resolution_correction = compute_resolution_correction(
            resolution_correction_method,
            wavenumber,
            real_space_delta.pixel_step,
            real_space_delta.binning,
            resolution_matrix=real_space_delta.resomat,
            mean_resolution=real_space_delta.metadata["MEANRESO"],
            pixelization_correction=False,
        )

        additional_fields = {
            "delta_noise": fourier_delta_noise,
            "delta_diff": fourier_delta_diff,
            "resolution_correction": resolution_correction,
        }
        return cls(wavenumber=wavenumber, delta=fourier_delta, **additional_fields)

    def plot_delta(
        self,
        ax=None,
    ):

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.wavenumber, self.delta)
        return ax


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
        # CR - In this implementation, log10 binning assumes km.s^-1 units.
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
