import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

from lyaps.constants import SPEED_LIGHT
from lyaps.delta import Delta, _available_binnings


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

        # if delta_arguments.get("ivar") is not None:
        #     delta_noise = fourier_delta * np.sqrt(delta_arguments["ivar"])

        # if delta_arguments.get("diff") is not None:
        #     diff_noise = fourier_delta * np.sqrt(delta_arguments["diff"])

        return cls(wavenumber=wavenumber, delta=fourier_delta)

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
