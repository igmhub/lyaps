import fitsio

from lyaps.deltas import RealSpaceDelta
from lyaps.fourier import FourierSpaceDelta


def read_real_space_delta_from_file(delta_file):
    """Reads a delta object from a file."""
    delta_list = []
    with fitsio.FITS(delta_file) as f:
        for data_unit in f:
            delta_list.append(RealSpaceDelta.from_fitsio(data_unit))
    return delta_list


def read_fourier_space_delta_from_file(fourier_delta_file):
    fourier_delta_list = []
    with fitsio.FITS(fourier_delta_file) as f:
        for data_unit in f:
            fourier_delta_list.append(FourierSpaceDelta.from_fitsio(data_unit))
    return fourier_delta_list


# + writing functions
