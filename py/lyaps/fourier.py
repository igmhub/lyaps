# Move all classes in a io spot
# Or put the functions inside the classes, to check

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq

from lyaps.constants import SPEED_LIGHT

_available_binnings = ["linear", "log10"]


class Delta(object):
    def __init__(
        self,
        delta=None,
        metadata=None,
        **kwargs,
    ):
        self.delta = delta
        self.metadata = metadata
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_fitsio(cls, data_unit):
        data_table = data_unit.read()
        metadata = data_unit.read_header()
        array_to_load = {}
        delta = None
        for _, data in enumerate(data_table.dtype.names):
            # CR - blind not treated here, to change that, modify here
            if "delta" in data.lower():
                delta = data_table[data]
            elif data.lower() == "lambda":
                array_to_load["wavelength"] = data_table[data]
            elif data.lower() == "log_lambda":
                array_to_load["log_wavelength"] = data_table[data]
            else:
                array_to_load[data.lower()] = data_table[data]

        return cls(
            delta=delta,
            metadata=metadata,
            **array_to_load,
        )


class RealSpaceDelta(Delta):
    def __init__(
        self,
        delta=None,
        metadata=None,
        wavelength=None,
        **kwargs,
    ):
        self.wavelength = wavelength
        self.wavelength_weights = np.ones_like(self.wavelength)
        super(RealSpaceDelta, self).__init__(delta=delta, metadata=metadata, **kwargs)
        self.set_binning()

    def set_binning(self):
        if hasattr(self, "wavelength"):
            self.binning, self.pixel_step = check_wavelenght_binning(self.wavelength)
        elif hasattr(self, "log_wavelength"):
            self.binning, self.pixel_step = check_wavelenght_binning(
                self.log_wavelength, log_wavelength_tolerance=1e-6
            )
        else:
            raise ValueError("No wavelength or log_wavelength attribute found")

    def fill_masked_pixels(
        self,
    ):
        (
            self.wavelength,
            self.delta,
            self.exposures_diff,
            self.wavelength_weights,
            self.ivar,
            self.number_masked_pixels,
        ) = fill_masked_pixels(
            self.pixel_step,
            self.wavelength,
            self.delta,
            self.wavelength_weights,
            self.ivar,
            # exposures_diff=self.exposures_diff,
        )

    def zeropad_to_grid(
        self,
        wavelength_grid,
        weight_array=None,
    ):
        self.delta, weight_grid = zeropad_to_grid(
            self.delta,
            self.wavelength,
            lambda_min_pad,
            lambda_max_pad,
            lambda_min_grid,
            lambda_max_grid,
            self.pixel_step,
            weight_array=self.weight_array,
            weight_array=None,
        )

        zeropad_to_grid(
            wavelength_grid,
            weight_array=weight_array,
        )
        self.wavelength = wavelength_grid
        self.weights = weight_grid
        if hasattr(self, "ivar"):
            self.ivar = np.zeros_like(self.wavelength)
        if hasattr(self, "exposures_diff"):
            self.exposures_diff = np.zeros_like(self.wavelength)

    def plot_delta(
        self,
        ax=None,
    ):

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.wavelength, self.delta)
        return ax


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


def check_wavelenght_binning(
    wavelength,
    wavelength_tolerance=1e-6,
    log_wavelength_tolerance=1e-6,
):
    diff_lambda = np.diff(wavelength)
    diff_log_lambda = np.diff(np.log10(wavelength))
    q5_lambda, q25_lambda = np.percentile(diff_lambda, [5, 25])
    q5_log_lambda, q25_log_lambda = np.percentile(diff_log_lambda, [5, 25])
    if (q25_lambda - q5_lambda) < wavelength_tolerance:
        binning = "linear"
        pixel_step = np.min(diff_lambda)
    elif (q25_log_lambda - q5_log_lambda) < log_wavelength_tolerance:
        binning = "log10"
        pixel_step = np.min(diff_log_lambda)
    else:
        raise ValueError(
            "Could not figure out if linear or log10 wavelength binning was used"
        )

    return binning, pixel_step


def fill_masked_pixels(
    pixel_step,
    wavelength,
    delta_array,
    weight_array,
    ivar,
    exposures_diff=None,
):
    wavelength_index = wavelength.copy()
    wavelength_index -= wavelength[0]
    wavelength_index /= pixel_step
    wavelength_index += 0.5
    wavelength_index = np.array(wavelength_index, dtype=int)
    index_all = range(wavelength_index[-1] + 1)
    index_ok = np.in1d(index_all, wavelength_index)

    delta_new = np.zeros(len(index_all))
    delta_new[index_ok] = delta_array

    wavelength_new = np.array(index_all, dtype=float)
    wavelength_new *= pixel_step
    wavelength_new += wavelength[0]

    exposures_diff_new = np.zeros(len(index_all))
    if exposures_diff is not None:
        exposures_diff_new[index_ok] = exposures_diff

    ivar_new = np.zeros(len(index_all), dtype=float)
    ivar_new[index_ok] = ivar

    weight_new = np.zeros(len(index_all), dtype=float)
    weight_new[index_ok] = weight_array

    num_masked_pixels = len(index_all) - len(wavelength_index)

    return (
        wavelength_new,
        delta_new,
        exposures_diff_new,
        ivar_new,
        num_masked_pixels,
    )


def zeropad_to_grid(
    delta_array,
    wavelength,
    lambda_min_pad,
    lambda_max_pad,
    lambda_min_grid,
    lambda_max_grid,
    pixel_step,
    weight_array=None,
):

    wavelength_grid = np.arange(lambda_min_grid, lambda_max_grid, pixel_step)
    delta_grid = np.zeros_like(wavelength_grid)
    weight_grid = np.zeros_like(wavelength_grid)

    mask_padding = (wavelength >= lambda_min_pad) & (wavelength < lambda_max_pad)
    mask_padding_grid = (wavelength_grid >= lambda_min_pad) & (
        wavelength_grid < lambda_max_pad
    )
    delta_grid[mask_padding_grid] = delta_array[mask_padding]
    if weight_array is not None:
        weight_grid = np.zeros_like(wavelength_grid)
        weight_grid[mask_padding_grid] = weight_array[mask_padding]
    else:
        weight_grid[mask_padding_grid] = 1.0
    return wavelength_grid, delta_grid, weight_grid

    # wavelength_grid = wavelength[mask_grid]
    # delta_grid = delta_array[mask_grid]

    # index_min_pad = round((wavelength - lambda_min_grid) / pixel_step)
    # index_max_pad = round((wavelength - lambda_max_grid,) / pixel_step)

    # wavelength_padded = np.pad(
    #     wavelength, (index_min_pad, index_max_pad), "constant", constant_values=(0.0)
    # )
    # return wavelength_padded


def zeropad_to_grid(
    delta_array,
    wavelength,
    wavelength_grid,
    pixel_step,
    weight_array=None,
):

    delta_grid = np.zeros_like(wavelength_grid)
    weight_grid = np.zeros_like(wavelength_grid)
    number_pixel_grid = len(wavelength_grid)

    index_min_data = round((wavelength[0] - wavelength_grid[0]) / pixel_step)
    index_max_data = round((wavelength[-1] - wavelength_grid[0]) / pixel_step)

    loz_cut = False
    hiz_cut = False
    if index_min_data < 0:
        loz_cut = True
        if index_max_data >= 0:
            delta_grid[:index_max_data] = delta_array[-index_min_data + 1 :]
            if weight_array is not None:
                weight_grid[:index_max_data] = weight_array[-index_min_data + 1 :]
            else:
                weight_grid[:index_max_data] = 1.0
    if index_max_data >= number_pixel_grid:
        hiz_cut = True
        if index_min_data < number_pixel_grid:
            delta_grid[index_min_data:] = delta_array[
                : number_pixel_grid - index_max_data - 1
            ]
            if weight_array is not None:
                weight_grid[index_min_data:] = weight_array[
                    : number_pixel_grid - index_max_data - 1
                ]
            else:
                weight_grid[index_min_data:] = 1.0
    if loz_cut == False and hiz_cut == False:
        delta_grid[index_min_data : index_max_data + 1] = delta_array
        if weight_array is not None:
            weight_grid[index_min_data : index_max_data + 1] = weight_array
        else:
            weight_grid[index_min_data : index_max_data + 1] = 1.0

    # CR - add masking weights

    return delta_grid, weight_grid


def split_forest():
    # Create several chunks of the delta
    return
