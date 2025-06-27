import glob
import itertools
import logging
import multiprocessing as mp
import os
from functools import partial

from lyaps.config import Config
from lyaps.deltas import treat_one_delta_object
from lyaps.fourier import fourier_transform_one_delta_object, group_fourier_chunks
from lyaps.io import read_all_real_space_delta_files, write_all_chunk_file


def process_all_separated_steps(config_filename):
    config = Config(config_filename)
    log = logging.getLogger("log")
    delta_filenames = glob.glob(
        os.path.join(config.in_dir, config.general_config.getstr("string search"))
    )
    reading_mode = config.general_config.getstr("reading mode")
    number_process = config.general_config.getint("num processors")
    grouping_method = config.general_config.getstr("grouping output method")
    overwrite = config.general_config.getstr("overwrite")

    general_config_dict = config.dict_params("general")
    delta_config_dict = config.dict_params("delta")
    fourier_config_dict = config.dict_params("fourier")

    function = partial(
        compute_fourier_chunks,
        delta_config_dict=delta_config_dict,
        general_config_dict=general_config_dict,
        fourier_config_dict=fourier_config_dict,
    )

    log.info(f"Reading delta files from {delta_filenames} in {reading_mode} mode.")

    deltas = read_all_real_space_delta_files(
        delta_filenames,
        reading_mode,
        number_process,
    )

    log.info(f"Read {len(deltas)} delta objects.")

    log.info("Treating delta objects to create chunks and fourier transform them.")

    if number_process == 1:
        chunk_list = []
        for delta_object in deltas:
            fourier_chunk_list = function(delta_object)
            chunk_list.extend(fourier_chunk_list)
    else:
        with mp.Pool(number_process) as pool:
            output_pool = pool.map(
                function,
                deltas,
            )
        chunk_list = list(itertools.chain.from_iterable(output_pool))

    log.info(f"Transformed {len(fourier_chunk_list)} chunks to Fourier space.")

    log.info(f"Grouping chunks in Fourier space with method {grouping_method}.")

    chunk_list_grouped = group_fourier_chunks(
        chunk_list,
        grouping_method,
    )

    log.info("Writing Fourier chunks to disk.")

    write_all_chunk_file(
        os.path.join(config.out_dir, "FourierDelta"),
        chunk_list_grouped,
        overwrite,
        number_process,
    )


def compute_fourier_chunks(
    delta_object,
    delta_config_dict,
    general_config_dict,
    fourier_config_dict,
):

    chunks = treat_one_delta_object(
        delta_object,
        delta_config_dict,
        general_config_dict,
    )

    fourier_chunk_list = []
    for chunk_id, chunk in enumerate(chunks):
        fourier_chunk_list.append(
            fourier_transform_one_delta_object(
                chunk,
                fourier_config_dict,
                chunk_id=chunk_id,
            )
        )

    return fourier_chunk_list
