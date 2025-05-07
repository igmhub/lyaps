import glob
from pathlib import Path

import git
from git import InvalidGitRepositoryError
from setuptools import find_namespace_packages, setup

scripts = sorted(glob.glob("bin/lyaps*"))

exec(open("py/lyaps/_version.py").read())
version = __version__

try:
    description = (
        f"Package for Igm Cosmological-Correlations Analyses\n"
        f"commit hash: {git.Repo('.').head.object.hexsha}"
    )
except InvalidGitRepositoryError:
    description = (
        f"Package for Igm Cosmological-Correlations Analyses\n" f"version: {version}"
    )
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="lyaps",
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igmhub/lyaps",
    author="Corentin Ravoux, the DESI Lya forest software topical group, et al",
    author_email="corentin.ravoux.research@gmail.com",
    packages=find_namespace_packages(where="py"),
    package_dir={"": "py"},
    package_data={"lyaps": []},
    install_requires=[
        "numpy",
        "scipy",
        "fitsio",
    ],
    scripts=scripts,
)
