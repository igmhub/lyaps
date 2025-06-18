import matplotlib.pyplot as plt
import numpy as np

from lyaps.constants import ABSORBER_IGM

_available_binnings = ["linear", "log10"]


# Add fucntion for cutting lambda min and max in the forest (observed wavelength)
# Add function for removing a forest with a too small number of pixels


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

    def return_editable_arrays_dict(
        self,
        keys_removed=[],
    ):
        editable_arrays = {}
        for key, array in self.__dict__.items():
            if key not in keys_removed and key not in ["delta"]:
                if isinstance(array, np.ndarray) and array.size == self.delta.size:
                    editable_arrays[key] = array
        return editable_arrays

    def return_editable_matrices_dict(
        self,
        keys_removed=[],
    ):
        editable_matrices = {}
        for key, array in self.__dict__.items():
            if key not in keys_removed:
                if (
                    isinstance(array, np.ndarray)
                    and array.ndim >= 2
                    and array.shape[0] == self.delta.size
                ):
                    editable_matrices[key] = array
        return editable_matrices

    def replace_arrays(
        self,
        new_arrays,
    ):
        for key, array in new_arrays.items():
            setattr(self, key, array)

    def replace_matrices(
        self,
        new_matrices,
    ):
        for key, matrix in new_matrices.items():
            setattr(self, key, matrix)


class RealSpaceDelta(Delta):
    def __init__(
        self,
        delta=None,
        metadata=None,
        wavelength=None,
        **kwargs,
    ):
        self.wavelength = wavelength
        self.number_masked_pixels = None
        super(RealSpaceDelta, self).__init__(delta=delta, metadata=metadata, **kwargs)
        self.set_binning()

    @classmethod
    def from_delta_zero_padding(
        cls,
        delta,
        wavelength_grid,
        factor_padding,
    ):

        other_arrays = delta.return_editable_arrays_dict(
            keys_removed=["wavelength", "log_wavelength"],
        )
        other_matrices = delta.return_editable_matrices_dict()
        padded_wavelength, padded_delta, padded_other_arrays, padded_other_matrices = (
            zero_padding(
                delta.delta,
                delta.wavelength,
                wavelength_grid,
                factor_padding,
                delta.pixel_step,
                other_arrays=other_arrays,
                other_matrices=other_matrices,
            )
        )

        padded_delta = cls(
            delta=padded_delta,
            metadata=delta.metadata,
            wavelength=padded_wavelength,
        )
        padded_delta.replace_arrays(padded_other_arrays)
        padded_delta.replace_matrices(padded_other_matrices)

        return padded_delta

    def set_binning(self):
        if hasattr(self, "wavelength"):
            self.binning, self.pixel_step = get_wavelenght_binning(self.wavelength)
        elif hasattr(self, "log_wavelength"):
            self.binning, self.pixel_step = get_wavelenght_binning(
                self.log_wavelength, log_wavelength_tolerance=1e-6
            )
        else:
            raise ValueError("No wavelength or log_wavelength attribute found")

    def set_mean_redshift(self, absorber_igm="LYA"):
        lambda_abs_igm = ABSORBER_IGM[absorber_igm]
        if self.binning == "linear":
            self.mean_redshift = np.mean(self.wavelength) / lambda_abs_igm - 1.0
        elif self.binning == "log10":
            self.mean_redshift = np.mean(10**self.log_wavelength) / lambda_abs_igm - 1.0
        else:
            raise ValueError(
                f"Binning must be {_available_binnings}, got {self.binning}"
            )

    def cut_wavelength(
        self,
        wavelength_min=None,
        wavelength_max=None,
    ):
        if wavelength_min is not None:
            mask = self.wavelength >= wavelength_min
            self.wavelength = self.wavelength[mask]
            self.delta = self.delta[mask]
            for key, array in self.return_editable_arrays_dict().items():
                setattr(self, key, array[mask])
            for key, matrix in self.return_editable_matrices_dict().items():
                setattr(self, key, matrix[mask, :])

        if wavelength_max is not None:
            mask = self.wavelength <= wavelength_max
            self.wavelength = self.wavelength[mask]
            self.delta = self.delta[mask]
            for key, array in self.return_editable_arrays_dict().items():
                setattr(self, key, array[mask])
            for key, matrix in self.return_editable_matrices_dict().items():
                setattr(self, key, matrix[mask, :])

    def fill_masked_pixels(
        self,
    ):
        if self.number_masked_pixels is None:
            (
                self.wavelength,
                self.delta,
                other_arrays_filled,
                other_matrices_filled,
                self.number_masked_pixels,
            ) = fill_masked_pixels(
                self.wavelength,
                self.delta,
                self.pixel_step,
                other_arrays=self.return_editable_arrays_dict(
                    keys_removed=["wavelength", "log_wavelength"],
                ),
                other_matrices=self.return_editable_matrices_dict(),
            )
            self.replace_arrays(other_arrays_filled)
            self.replace_matrices(other_matrices_filled)

    def plot_delta(
        self,
        ax=None,
    ):

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.wavelength, self.delta)
        legend = ["Delta"]
        if hasattr(self, "ivar"):
            mask = self.ivar > 0
            standard_deviation = np.zeros_like(self.ivar)
            standard_deviation[mask] = np.sqrt(1 / self.ivar[mask])
            standard_deviation[~mask] = np.nan
            ax.plot(self.wavelength, standard_deviation)
            legend.append("Noise")
        if hasattr(self, "weight"):
            ax.plot(self.wavelength, self.weight)
            legend.append("Weight")
        ax.legend(legend)
        ax.set_xlabel("Wavelength")


def get_wavelenght_binning(
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
    wavelength,
    delta_array,
    pixel_step,
    other_arrays=None,
    other_matrices=None,
):
    wavelength_index = wavelength.copy()
    wavelength_index -= wavelength[0]
    wavelength_index /= pixel_step
    wavelength_index += 0.5
    wavelength_index = np.array(wavelength_index, dtype=int)
    index_all = range(wavelength_index[-1] + 1)
    index_ok = np.in1d(index_all, wavelength_index)

    delta_array_filled = np.zeros(len(index_all))
    delta_array_filled[index_ok] = delta_array

    wavelength_filled = np.array(index_all, dtype=float)
    wavelength_filled *= pixel_step
    wavelength_filled += wavelength[0]

    other_arrays_filled = {}
    if other_arrays is not None:
        for key, other_array in other_arrays.items():
            other_array_filled = np.zeros(len(index_all))
            other_array_filled[index_ok] = other_array
            other_arrays_filled[key] = other_array_filled

    other_matrices_filled = {}
    if other_matrices is not None:
        for key, other_matrix in other_matrices.items():
            other_matrix_filled = np.zeros((len(index_all), other_matrix.shape[1]))
            other_matrix_filled[index_ok, :] = other_matrix
            other_matrices_filled[key] = other_matrix_filled

    num_masked_pixels = len(index_all) - len(wavelength_index)

    return (
        wavelength_filled,
        delta_array_filled,
        other_arrays_filled,
        other_matrices_filled,
        num_masked_pixels,
    )


def zero_padding(
    delta_array,
    wavelength,
    wavelength_grid,
    factor_padding,
    pixel_step,
    other_arrays=None,
    other_matrices=None,
    tolerance_missalignement=10**-4,
):
    if factor_padding < 2:
        raise ValueError("Factor padding must be greater than 2")

    mask_in = wavelength > wavelength_grid[0] - pixel_step / 2
    mask_in &= wavelength < wavelength_grid[-1] + pixel_step / 2

    if np.sum(mask_in) == 0:
        return None, None, None

    mask_in_grid = wavelength_grid > wavelength[0] - pixel_step / 2
    mask_in_grid &= wavelength_grid < wavelength[-1] + pixel_step / 2

    if np.any(
        np.abs(wavelength_grid[mask_in_grid] - wavelength[mask_in])
        > tolerance_missalignement
    ):
        raise ValueError(
            "Wavelength grid does not match the wavelength array within the tolerance."
        )

    size_before = (wavelength_grid < wavelength[0] - pixel_step / 2).sum()
    size_after = (wavelength_grid > wavelength[-1] + pixel_step / 2).sum()

    size_padding = int((factor_padding - 1) * len(wavelength_grid) / 2)

    padded_wavelength = np.concatenate(
        [
            np.arange(
                wavelength_grid[0] - (size_padding + 1) * pixel_step,
                wavelength_grid[0] - pixel_step,
                pixel_step,
            ),
            wavelength_grid,
            np.arange(
                wavelength_grid[-1] + pixel_step,
                wavelength_grid[-1] + (size_padding + 1) * pixel_step,
                pixel_step,
            ),
        ]
    )

    padded_array = np.zeros(size_padding)

    delta_grid = np.concatenate(
        [
            np.zeros(size_before),
            delta_array[mask_in],
            np.zeros(size_after),
        ]
    )
    padded_delta = np.concatenate(
        [
            padded_array,
            delta_grid,
            padded_array,
        ]
    )
    padded_other_arrays = {}
    if other_arrays is not None:
        for key, other_array in other_arrays.items():

            other_array_grid = np.concatenate(
                [
                    np.zeros(size_before),
                    other_array[mask_in],
                    np.zeros(size_after),
                ]
            )
            padded_other_arrays[key] = np.concatenate(
                [
                    padded_array,
                    other_array_grid,
                    padded_array,
                ]
            )

    padded_other_matrices = {}
    if other_matrices is not None:
        for key, other_matrix in other_matrices.items():
            padded_matrix = np.transpose(
                np.tile(padded_array, (other_matrix.shape[1], 1))
            )

            other_matrix_grid = np.concatenate(
                (
                    np.zeros((size_before, other_matrix.shape[1])),
                    other_matrix[mask_in, :],
                    np.zeros((size_after, other_matrix.shape[1])),
                ),
                axis=0,
            )
            padded_other_matrices[key] = np.concatenate(
                (
                    padded_matrix,
                    other_matrix_grid,
                    padded_matrix,
                ),
                axis=0,
            )

    return padded_wavelength, padded_delta, padded_other_arrays, padded_other_matrices


def split_forest(
    delta_array,
    wavelength,
    number_parts,
    pixel_step,
    other_arrays=None,
    other_matrices=None,
):

    num_bins = len(wavelength) // number_parts

    delta_array_chunks = []
    wavelength_chunks = []
    other_arrays_chunks = []
    other_matrices_chunks = []

    for index in range(number_parts):
        if index == 0:
            wavelength_limit_min = wavelength[0] - 0.5 * pixel_step
        else:
            wavelength_limit_min = wavelength[num_bins * index]
        if index == number_parts - 1:
            wavelength_limit_max = wavelength[len(wavelength) - 1] + 0.5 * pixel_step
        else:
            wavelength_limit_max = wavelength[num_bins * (index + 1)]

        mask = (wavelength > wavelength_limit_min) & (wavelength < wavelength_limit_max)

        wavelength_chunks.append(wavelength[mask])
        delta_array_chunks.append(delta_array[mask])
        if other_arrays is not None:
            other_arrays_chunk_dict = {}
            for key, other_array in other_arrays.items():
                other_arrays_chunk_dict[key] = other_array[mask]
            other_arrays_chunks.append(other_arrays_chunk_dict)

        if other_matrices is not None:
            other_matrices_chunk_dict = {}
            for key, other_matrix in other_matrices.items():
                other_matrices_chunk_dict[key] = other_matrix[mask, :]
            other_matrices_chunks.append(other_matrices_chunk_dict)

    return (
        wavelength_chunks,
        delta_array_chunks,
        other_arrays_chunks,
        other_matrices_chunks,
    )


def split_forest_in_redshift_parts(
    delta_array,
    wavelength,
    z_grid,
    other_arrays=None,
    other_matrices=None,
    absorber_igm="LYA",
    min_num_pixels=None,
):

    wavelength_grid_min = ABSORBER_IGM[absorber_igm] * (1 + z_grid[:-1])
    wavelength_grid_max = ABSORBER_IGM[absorber_igm] * (1 + z_grid[1:])

    number_parts = 0
    delta_array_chunks = []
    wavelength_chunks = []
    other_arrays_chunks = []
    other_matrices_chunks = []

    for wavelength_chunk_min, wavelength_chunk_max in zip(
        wavelength_grid_min, wavelength_grid_max
    ):
        mask = (wavelength >= wavelength_chunk_min) & (
            wavelength < wavelength_chunk_max
        )
        if min_num_pixels is not None and np.sum(mask) < min_num_pixels:
            continue
        if (
            np.sum(wavelength <= wavelength_chunk_min) == 0
            or np.sum(wavelength >= wavelength_chunk_max) == 0
        ):
            continue
        number_parts += 1
        wavelength_chunks.append(wavelength[mask])
        delta_array_chunks.append(delta_array[mask])
        if other_arrays is not None:
            other_arrays_chunk_dict = {}
            for key, other_array in other_arrays.items():
                other_arrays_chunk_dict[key] = other_array[mask]
            other_arrays_chunks.append(other_arrays_chunk_dict)

        if other_matrices is not None:
            other_matrices_chunk_dict = {}
            for key, other_matrix in other_matrices.items():
                other_matrices_chunk_dict[key] = other_matrix[mask, :]
            other_matrices_chunks.append(other_matrices_chunk_dict)

    return (
        wavelength_chunks,
        delta_array_chunks,
        other_arrays_chunks,
        other_matrices_chunks,
        number_parts,
    )


def create_deltas_from_split(
    delta_object,
    cut_in_redshift=False,
    number_parts=None,
    z_grid=None,
):
    if not cut_in_redshift and number_parts is None:
        raise ValueError("If cut_in_redshift is False, number_parts must be provided.")
    if cut_in_redshift and z_grid is None:
        raise ValueError("z_grid must be provided when cut_in_redshift is True.")
    if cut_in_redshift:

        (
            wavelength_chunks,
            delta_array_chunks,
            other_arrays_chunks,
            other_matrices_chunks,
            number_parts,
        ) = split_forest_in_redshift_parts(
            delta_object.delta,
            delta_object.wavelength,
            z_grid,
            other_arrays=delta_object.return_editable_arrays_dict(
                keys_removed=["wavelength", "log_wavelength"],
            ),
            other_matrices=delta_object.return_editable_matrices_dict(),
        )
    else:
        (
            wavelength_chunks,
            delta_array_chunks,
            other_arrays_chunks,
            other_matrices_chunks,
        ) = split_forest(
            delta_object.delta,
            delta_object.wavelength,
            number_parts,
            delta_object.pixel_step,
            other_arrays=delta_object.return_editable_arrays_dict(
                keys_removed=["wavelength", "log_wavelength"],
            ),
            other_matrices=delta_object.return_editable_matrices_dict(),
        )

    splitted_delta_array = []
    for index in range(number_parts):

        splitted_delta = RealSpaceDelta(
            delta=delta_array_chunks[index],
            metadata=delta_object.metadata,
            wavelength=wavelength_chunks[index],
        )
        splitted_delta.replace_arrays(other_arrays_chunks[index])
        splitted_delta.replace_matrices(other_matrices_chunks[index])
        splitted_delta_array.append(splitted_delta)

    return splitted_delta_array
