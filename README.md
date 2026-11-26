# lyaps

[![tests](https://github.com/igmhub/lyaps/actions/workflows/tests.yml/badge.svg)](https://github.com/igmhub/lyaps/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Python package for power spectrum calculation with Lyman-alpha forests.

## What it does

lyaps measures the one-dimensional Lyman-alpha forest flux power spectrum
(P1D) with an FFT estimator. It reads delta files (picca format), cuts,
fills masked pixels and splits or zero-pads each forest into chunks, then
Fourier transforms each chunk with its noise power and resolution
correction, and writes the Fourier-space chunks grouped by file or redshift.
`lyaps.fft1d` averages per-chunk P1D files into a mean P1D with covariance.

```
delta files ─► lyaps.io ─► lyaps.deltas ─► lyaps.fourier ─► fourier-delta-*.fits.gz
                (read)     (cut, fill,      (FFT, noise,
                            split / pad)     resolution)
```

## Install

```bash
git clone https://github.com/igmhub/lyaps
cd lyaps
pip install -e .
```

## Quick start

```python
from lyaps.main import process_all_separated_steps

config = {
    "general": {
        "in dir": "/path/to/Delta",
        "out dir": "/path/to/output",
        "overwrite": True,
        "reading mode": "file",
        "num processors": 1,
        "grouping output method": "redshift",
    },
    "delta": {
        "wavelength weight definition": "ivar",
        "fill masked pixels": True,
        "split forest": False,
        "zero padding": True,
        "split in redshift": False,
    },
    "fourier": {"resolution correction method": "matrix"},
}
process_all_separated_steps(config)  # a dict, or the path to a .ini file
```

## Release

```bash
uvx bump-my-version bump patch   # or minor / major: commits + tags v<version>
git push --follow-tags
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

GPL-3.0 — see [`LICENSE`](LICENSE).
