import logging
import os
from configparser import ConfigParser


class ConfigError(Exception):
    """
    Exceptions occurred in class Config
    """


accepted_general_options = [
    "in dir",
    "out dir",
    "overwrite",
    "string search",
    "reading mode",
    "absorber igm",
    "log",
    "logging level console",
    "logging level file",
    "num processors",
    "grouping output method",
]

mandatory_general_options = [
    "in dir",
    "out dir",
    "overwrite",
]

accepted_delta_options = [
    "fill masked pixels",
    "split forest",
    "zero padding",
    "split in redshift",
    "wavelength weight definition",
    "wavelength observed minimum",
    "wavelength observed maximum",
    "number parts",
    "redshift min split",
    "redshift max split",
    "redshift step split",
    "factor padding",
    "telescop wavelength grid",
    "redshift min padding",
    "redshift max padding",
    "redshift step padding",
    "minimal chunks size",
    "maximal masked pixels",
    "minimum snr",
    "maximal mean resolution",
]

mandatory_delta_options = [
    "fill masked pixels",
    "split forest",
    "zero padding",
    "split in redshift",
]

accepted_fourier_options = [
    "resolution correction method",
    "pixelization correction",
    "number noise realization",
    "white noise assumption",
    "rebin diff noise",
]

mandatory_fourier_options = []


default_config = {
    "general": {
        "string search": "delta*.fits*",
        "absorber igm": "LYA",
        "reading mode": "individual",
        "log": "run.log",
        "logging level console": "DEBUG",
        "logging level file": "DEBUG",
        "num processors": 1,
    },
    "delta": {
        "redshift min split": 2.1,
        "redshift max split": 3.2,
        "redshift step split": 0.2,
        "factor padding": 2,
        "telescop wavelength grid": "desi",
        "redshift min padding": 2.1,
        "redshift max padding": 3.2,
        "redshift step padding": 0.2,
    },
    "fourier": {
        "resolution correction method": "matrix",
        "pixelization correction": False,
        "number noise realization": 200,
        "white noise assumption": True,
        "rebin diff noise": False,
    },
}


def check_options(section, accepted_options, section_name):
    for key in section.keys():
        if key not in accepted_options:
            raise ConfigError(
                f"Unrecognised option in section [{section_name}]."
                f"Found: '{key}'. Accepted options are "
                f"{accepted_options}"
            )


def check_mandatory_options(section, mandatory_options, section_name):
    for key in mandatory_options:
        if key not in section.keys():
            raise ConfigError(
                f"Missing variable '{key}' in section [{section_name}]."
                f"Needed options are "
                f"{mandatory_options}"
            )


def parse_float(input):
    if input == "None":
        return None
    else:
        return float(input)


def parse_int(input):
    if input == "None":
        return None
    else:
        return int(input)


def parse_string(input):
    if input == "None":
        return None
    else:
        return str(input)


def parse_param(input):
    if input == "None":
        return None
    else:
        try:
            return int(input)
        except:
            try:
                return float(input)
            except:
                if input.lower() == "true":
                    return True
                elif input.lower() == "false":
                    return False
                else:
                    return str(input)


class Config(object):

    def __init__(self, filename):

        self.config = ConfigParser(
            allow_no_value=True,
            converters={
                "str": parse_string,
                "int": parse_int,
                "float": parse_float,
                "param": parse_param,
            },
        )
        self.config.optionxform = lambda option: option
        self.config.read_dict(default_config)

        if type(filename) is dict:
            self.config.read_dict(filename)
        elif type(filename) is str:
            if os.path.isfile(filename):
                self.config.read(filename)
            else:
                raise ConfigError(f"Config file not found: {filename}")
        else:
            raise ConfigError(
                "Config file should be a dictionary or a string with the "
                "path to the configuration file."
            )

        self.__parse_environ_variables()
        self.__format_section("general")
        self.__format_section("delta")
        self.__format_section("fourier")
        self.__setup_io()
        self.__initialize_folders()
        self.__setup_logging()

    def __format_section(
        self,
        section_name,
    ):
        if section_name not in self.config:
            raise ConfigError(f"Missing section [{section_name}]")
        section = self.config[section_name]
        check_options(section, eval(f"accepted_{section_name}_options"), section_name)
        check_mandatory_options(
            section, eval(f"mandatory_{section_name}_options"), section_name
        )
        setattr(self, f"{section_name}_config", section)

    def __parse_environ_variables(self):
        """Read all variables and replaces the enviroment variables for their
        actual values. This assumes that enviroment variables are only used
        at the beggining of the paths.

        Raise
        -----
        ConfigError if an environ variable was not defined
        """
        for section in self.config:
            for key, value in self.config[section].items():
                if value.startswith("$"):
                    pos = value.find("/")
                    if os.getenv(value[1:pos]) is None:
                        raise ConfigError(
                            f"In section [{section}], undefined "
                            f"environment variable {value[1:pos]} "
                            "was found"
                        )
                    self.config[section][key] = value.replace(
                        value[:pos], os.getenv(value[1:pos])
                    )

    def __initialize_folders(self):

        if not os.path.exists(f"{self.out_dir}/.config.ini"):
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "FourierDelta"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "Log"), exist_ok=True)
            self.write_config()
        elif self.overwrite:
            os.makedirs(os.path.join(self.out_dir, "FourierDelta"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "Log"), exist_ok=True)
            self.write_config()
        else:
            raise ConfigError(
                "Specified folder contains a previous run. "
                "Pass overwrite option in configuration file "
                "in order to ignore the previous run or "
                "change the output path variable to point "
                f"elsewhere. Folder: {self.out_dir}"
            )

    def __setup_io(self):
        self.in_dir = self.config["general"].getstr("in dir")
        self.out_dir = self.config["general"].getstr("out dir")
        self.overwrite = self.config["general"].getboolean("overwrite")

    def __setup_logging(self):

        if "/" in self.config["general"].getstr("log"):
            raise ConfigError(
                "Variable 'log' in section [general] should not incude folders. "
                f"Found: {self.log}"
            )

        logging_level_file = self.config["general"].getstr("logging level file").upper()

        logging_level_console = (
            self.config["general"].getstr("logging level console").upper()
        )

        logging_file_name = os.path.join(
            self.out_dir,
            "Log",
            self.config["general"].getstr("log"),
        )

        log = logging.getLogger("log")
        filehandler = logging.FileHandler(
            logging_file_name,
            mode="a",
        )
        filehandler.setLevel(eval(f"logging.{logging_level_file}"))
        fileformatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        filehandler.setFormatter(fileformatter)
        filehandler.propagate = False
        log.addHandler(filehandler)

        consolehandler = logging.StreamHandler()
        consolehandler.setLevel(eval(f"logging.{logging_level_console}"))
        consolehandler.setLevel(logging.INFO)
        consoleformatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        consolehandler.setFormatter(consoleformatter)
        log.addHandler(consolehandler)

    def write_config(self):
        """This function writes the configuration options for later
        usages. The file is saved under the name .config.ini and in
        the self.out_dir folder
        """
        outname = f"{self.out_dir}/.config.ini"
        if os.path.exists(outname):
            newname = f"{outname}.{os.path.getmtime(outname)}"
            os.rename(outname, newname)
        with open(outname, "w", encoding="utf-8") as config_file:
            self.config.write(config_file)

    def dict_params(self, section_name):
        accepted_options = eval(f"accepted_{section_name}_options")
        dict_param = {}
        for key in accepted_options:
            param = self.config[section_name].getparam(key)
            dict_param[key] = param
        return dict_param
