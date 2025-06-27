import itertools
import multiprocessing as mp
import os
from functools import partial

import fitsio

from lyaps.deltas import RealSpaceDelta

_available_reading_mode = ["file", "hdu"]


def read_real_space_delta_from_file(delta_file):
    """Reads a delta object from a file."""
    delta_list = []
    with fitsio.FITS(delta_file) as f:
        for data_unit in f[1:]:
            delta_list.append(RealSpaceDelta.from_fitsio(data_unit))
    return delta_list


def read_hdu_data_from_file(filename):
    hdu_data = []
    with fitsio.FITS(filename) as f:
        for data_unit in f[1:]:
            metadata = data_unit.read_header()
            metadata["FILENAME"] = data_unit.get_filename().split("/")[-1]
            hdu_data.append((data_unit.read(), metadata))
    return hdu_data


def read_all_real_space_delta_files(
    delta_filenames,
    reading_mode,
    number_process,
):
    if reading_mode == "file":
        if number_process == 1:
            delta_list = []
            for _, filename in enumerate(delta_filenames):
                deltas = read_real_space_delta_from_file(filename)
                delta_list.extend(deltas)
        else:
            with mp.Pool(number_process) as pool:
                output_pool = pool.map(
                    read_real_space_delta_from_file,
                    delta_filenames,
                )
                delta_list = list(itertools.chain.from_iterable(output_pool))
    elif reading_mode == "hdu":
        if number_process == 1:
            delta_fits_data_list = []
            for filename in delta_filenames:
                delta_fits_data_list.extend(read_hdu_data_from_file(filename))
        else:
            with mp.Pool(number_process) as pool:
                output_pool = pool.map(
                    read_hdu_data_from_file,
                    delta_filenames,
                )
                delta_fits_data_list = list(itertools.chain.from_iterable(output_pool))

        if number_process == 1:
            delta_list = [
                RealSpaceDelta.from_fitsio_data(*fits_data)
                for fits_data in delta_fits_data_list
            ]
        else:
            with mp.Pool(number_process) as pool:
                delta_list = pool.starmap(
                    RealSpaceDelta.from_fitsio_data,
                    delta_fits_data_list,
                )
    else:
        raise ValueError(
            f"Unsupported reading mode: {reading_mode}"
            f"Supported modes are {_available_reading_mode}."
        )
    return delta_list


def write_one_chunk_group(chunk_group, output_path, overwrite):

    arg_filename_out, chunk_list = chunk_group[0], chunk_group[1]
    hdu_list = []
    filename_out = os.path.join(
        output_path, f"fourier-delta-{arg_filename_out}.fits.gz"
    )
    for _, chunk in enumerate(chunk_list):
        hdu_list.append(chunk.create_hdu())
    with fitsio.FITS(filename_out, "rw", clobber=overwrite) as f:
        for hdu in hdu_list:
            f.write(hdu[0], header=hdu[1])


def write_all_chunk_file(
    output_path,
    chunk_list_grouped,
    overwrite,
    number_process,
):

    function = partial(
        write_one_chunk_group,
        output_path=output_path,
        overwrite=overwrite,
    )

    if number_process == 1:
        for _, group in enumerate(chunk_list_grouped):
            function(group)

    else:
        with mp.Pool(number_process) as pool:
            pool.map(
                function,
                chunk_list_grouped,
            )


# + writing functions for FourierSpaceDelta objects
# + Reader for FourierSpaceDelta objects
