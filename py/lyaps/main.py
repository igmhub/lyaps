import logging
import os

from lyaps import io
from lyaps.config import Config
from lyaps.deltas import treat_one_delta_object
from lyaps.fourier import fourier_transform_one_delta_object


def fourier_transform_all_deltas(config_filename):
    config = Config(config_filename)
    log = logging.getLogger("log")

    delta_filenames = os.path.join(
        config.in_dir, config.general_config("string search")
    )
    reading_mode = config.general_config("reading mode")

    # Add some cuts
    # Fix wavelength cut

    log.info(f"Reading delta files from {delta_filenames} in {reading_mode} mode.")

    deltas = io.read_real_space_deltas(
        delta_filenames,
        reading_mode,
        config.general_config,
    )

    log.info(f"Read {len(deltas)} delta objects.")

    log.info("Treating delta objects to create chunks.")
    chunk_list = []
    for delta_object in deltas:

        chunks = deltas.treat_one_delta_object(
            delta_object,
            config.general_config,
            config.delta_config,
        )
        chunk_list = chunk_list + chunks
    log.info(f"Created {len(chunk_list)} chunks from delta objects.")
    log.info("Starting Fourier transform of chunks.")

    fourier_chunk_list = []
    for chunk in chunk_list:
        fourier_chunk_list.append(
            fourier_transform_one_delta_object(chunk, config["fourier"])
        )

    log.info(f"Transformed {len(fourier_chunk_list)} chunks to Fourier space.")

    # Grouping and writing the fourier chunks

    grouping_method = config.fourier_config("grouping method")

    log.info(f"Grouping chunks in Fourier space with method {grouping_method}.")
    log.info("Writing Fourier chunks to disk.")


# Add function for removing a forest with a too small number of pixels

# if delta.mean_snr <= args.SNR_min or delta.mean_reso >= args.reso_max:
#     continue


# if linear_binning:
#     max_num_pixels_forest_theoretical = (
#         (1180 - 1050) * (delta.z_qso + 1) / pixel_step
#     )  # currently no min/max restframe values defined, so hard coding for the moment
# else:
#     max_num_pixels_forest_theoretical = (
#         np.log(1180) - np.log10(1050)
#     ) / pixel_step
# # minimum number of pixel in forest
# if args.nb_pixel_min is not None:
#     min_num_pixels = args.nb_pixel_min
# else:
#     min_num_pixels = int(
#         args.nb_pixel_frac_min * max_num_pixels_forest_theoretical
#     )  # this is currently just hardcoding values so that spectra have a minimum length changing with z; but might be problematic for SBs
