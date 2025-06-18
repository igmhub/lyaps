def compute_pk_cross_exposure(fft_delta_array, fft_delta_array_2):
    """
    Computes the cross-exposure power spectrum estimator.

    Arguments
    ---------
    fft_delta_array: numpy.array
    Contains the the fft of delta for all individual exposures

    fft_delta_array_2: numpy.array
    Contains the the fft of delta for all individual exposures for the second term

    Return
    ------
    pk_cross_exposure: numpy.array
    The power spectrum of the cross exposure estimator

    """
    number_exposures = len(fft_delta_array)
    if number_exposures < 2:
        raise ValueError(
            "The cross exposure estimator can only be used "
            "with a number of exposures equal or higher than 2"
        )

    pk_cross_exposure = np.zeros_like(fft_delta_array[0], dtype=np.float_)

    for i, fft_delta_i in enumerate(fft_delta_array):
        for j, fft_delta_2_j in enumerate(fft_delta_array_2):
            if j != i:
                pk_cross_exposure += (
                    fft_delta_i.real * fft_delta_2_j.real
                    + fft_delta_i.imag * fft_delta_2_j.imag
                )
    number_pairs = number_exposures * (number_exposures - 1)
    pk_cross_exposure = pk_cross_exposure / number_pairs

    return pk_cross_exposure


class Pk1D:
    """Class to represent the 1D Power Spectrum for a given forest

    Class Methods
    -------------
    from_fitsio
    write_fits

    Methods
    -------
    __init__

    Attributes
    ----------
    ra: float
    Right-ascension of the quasar (in radians).

    dec: float
    Declination of the quasar (in radians).

    z_qso: float
    Redshift of the quasar.

    mean_z: float
    Mean redshift of the forest

    mean_snr: float
    Mean signal-to-noise ratio in the forest

    mean_reso: float
    Mean resolution of the forest

    num_masked_pixels: int
    Number of masked pixels

    linear_bining: bool
    If the spectra used is linearly binned or not

    los_id: int
    Indentifier of the quasar (TARGETID in DESI)

    chunk_id: int
    For forest cut in chunks, identifier of the chunk

    k: array of float
    Fourier modes

    pk_raw: array of float
    Raw power spectrum

    pk_noise: array of float
    Noise power spectrum for the different Fourier modes

    pk_diff: array of float or None - default: None
    Power spectrum of exposures_diff for the different Fourier modes

    correction_reso: array of float
    Resolution correction

    pk: array of float
    Power Spectrum

    fft_delta: array of float
    In the case where the fft of delta is stored, contains it

    fft_delta_noise: array of float
    fft of noise realization

    fft_delta_diff: array of float
    fft of exposure difference
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        ra=None,
        dec=None,
        z_qso=None,
        mean_z=None,
        mean_snr=None,
        mean_reso=None,
        num_masked_pixels=None,
        linear_bining=None,
        los_id=None,
        chunk_id=None,
        k=None,
        pk_raw=None,
        pk_noise=None,
        pk_diff=None,
        correction_reso=None,
        pk=None,
        fft_delta=None,
        fft_delta_noise=None,
        fft_delta_diff=None,
    ):
        """Initialize instance

        Arguments
        ---------
        ra: float
        Right-ascension of the quasar (in radians).

        dec: float
        Declination of the quasar (in radians).

        z_qso: float
        Redshift of the quasar.

        mean_z: float
        Mean redshift of the forest

        mean_snr: float
        Mean signal-to-noise ratio in the forest

        mean_reso: float
        Mean resolution of the forest

        num_masked_pixels: int
        Number of masked pixels

        linear_bining: bool
        If the spectra used is linearly binned or not

        los_id: int
        Indentifier of the quasar (TARGETID in DESI)

        chunk_id: int
        For forest cut in chunks, identifier of the chunk

        k: array of float
        Fourier modes

        pk_raw: array of float
        Raw power spectrum

        pk_noise: array of float
        Noise power spectrum for the different Fourier modes

        pk_diff: array of float or None - default: None
        Power spectrum of exposures_diff for the different Fourier modes

        correction_reso: array of float
        Resolution correction

        pk: array of float
        Power Spectrum

        fft_delta: array of float
        In the case where the fft of delta is stored, contains it

        fft_delta_noise: array of float
        fft of noise realization

        fft_delta_diff: array of float
        fft of exposure difference

        """
        self.ra = ra
        self.dec = dec
        self.z_qso = z_qso
        self.mean_z = mean_z
        self.mean_snr = mean_snr
        self.mean_reso = mean_reso
        self.num_masked_pixels = num_masked_pixels
        self.linear_bining = linear_bining
        self.los_id = los_id
        self.chunk_id = chunk_id

        self.k = k
        self.pk_raw = pk_raw
        self.pk_noise = pk_noise
        self.pk_diff = pk_diff
        self.correction_reso = correction_reso
        self.pk = pk
        self.fft_delta = fft_delta
        self.fft_delta_noise = fft_delta_noise
        self.fft_delta_diff = fft_delta_diff

    @classmethod
    def from_fitsio(cls, hdu):
        """Read the 1D Power Spectrum from fits file

        Arguments
        ---------
        hdu: Header Data Unit
        Header Data Unit where the 1D Power Spectrum is read

        Return
        ------
        An intialized instance of Pk1D
        """

        header = hdu.read_header()

        ra = header["RA"]
        dec = header["DEC"]
        z_qso = header["Z"]
        mean_z = header["MEANZ"]
        mean_reso = header["MEANRESO"]
        mean_snr = header["MEANSNR"]
        num_masked_pixels = header["NBMASKPIX"]
        linear_bining = header["LIN_BIN"]
        los_id = header["LOS_ID"]
        chunk_id = header["CHUNK_ID"]

        data = hdu.read()
        k = data["K"][:]
        pk_noise = data["PK_NOISE"][:]
        correction_reso = data["COR_RESO"][:]
        pk_diff = data["PK_DIFF"][:]
        pk = data["PK"][:]
        pk_raw = data["PK_RAW"][:]

        try:
            fft_delta = data["DELTA_K"][:]
            fft_delta_noise = data["DELTA_NOISE_K"][:]
            fft_delta_diff = data["DELTA_DIFF_K"][:]

        except ValueError:
            fft_delta = None
            fft_delta_noise = None
            fft_delta_diff = None

        return cls(
            ra=ra,
            dec=dec,
            z_qso=z_qso,
            mean_z=mean_z,
            mean_snr=mean_snr,
            mean_reso=mean_reso,
            num_masked_pixels=num_masked_pixels,
            linear_bining=linear_bining,
            los_id=los_id,
            chunk_id=chunk_id,
            k=k,
            pk_raw=pk_raw,
            pk_noise=pk_noise,
            pk_diff=pk_diff,
            correction_reso=correction_reso,
            pk=pk,
            fft_delta=fft_delta,
            fft_delta_noise=fft_delta_noise,
            fft_delta_diff=fft_delta_diff,
        )

    def write_fits(self, file):
        """Write an individual pk1D inside a FITS file.

        Arguments
        ---------
        file: fitsio file
        File to which a pk1D hdu will be added

        Return
        ------
        None
        """
        header = [
            {
                "name": "RA",
                "value": self.ra,
                "comment": "QSO's Right Ascension [degrees]",
            },
            {
                "name": "DEC",
                "value": self.dec,
                "comment": "QSO's Declination [degrees]",
            },
            {"name": "Z", "value": self.z_qso, "comment": "QSO's redshift"},
            {
                "name": "MEANZ",
                "value": self.mean_z,
                "comment": "Absorbers mean redshift",
            },
            {
                "name": "MEANSNR",
                "value": self.mean_snr,
                "comment": "Mean signal to noise ratio",
            },
            {
                "name": "MEANRESO",
                "value": self.mean_reso,
                "comment": "Mean resolution [km/s]",
            },
            {
                "name": "NBMASKPIX",
                "value": self.num_masked_pixels,
                "comment": "Number of masked pixels in the section",
            },
            {
                "name": "LIN_BIN",
                "value": self.linear_bining,
                "comment": "analysis was performed on delta with linear binned lambda",
            },
            {
                "name": "LOS_ID",
                "value": self.los_id,
                "comment": "line of sight identifier, e.g. THING_ID or TARGETID",
            },
            {
                "name": "CHUNK_ID",
                "value": self.chunk_id,
                "comment": "Chunk (sub-forest) identifier",
            },
        ]

        cols = [
            self.k,
            self.pk_raw,
            self.pk_noise,
            self.pk_diff,
            self.correction_reso,
            self.pk,
        ]
        names = ["K", "PK_RAW", "PK_NOISE", "PK_DIFF", "COR_RESO", "PK"]
        comments = [
            "Wavenumber",
            "Raw power spectrum",
            "Noise's power spectrum",
            "Noise coadd difference power spectrum",
            "Correction resolution function",
            "Corrected power spectrum (resolution and noise)",
        ]
        if self.linear_bining:
            baseunit = "AA"
        else:
            baseunit = "km/s"
        units = [
            f"({baseunit})^-1",
            f"{baseunit}",
            f"{baseunit}",
            f"{baseunit}",
            f"{baseunit}",
            f"{baseunit}",
        ]

        if self.fft_delta is not None:
            cols += [
                self.fft_delta,
                self.fft_delta_noise,
                self.fft_delta_diff,
            ]
            names += [
                "DELTA_K",
                "DELTA_NOISE_K",
                "DELTA_DIFF_K",
            ]
            comments += [
                "Fourier delta",
                "Fourier delta noise",
                "Fourier delta diff",
            ]
            units += [
                f"{baseunit}^(1/2)",
                f"{baseunit}^(1/2)",
                f"{baseunit}^(1/2)",
            ]

        file.write(
            cols,
            names=names,
            header=header,
            comment=comments,
            units=units,
        )
