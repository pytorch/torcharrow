# TorchArrow: a data processing library for PyTorch

**This library is currently in the Beta stage and does not have a stable release. The API and implementation may change based on
user feedback or performance. Future changes may not be backward compatible.
If you have suggestions on the API or use cases you'd like to be covered, please open a
GitHub issue. We'd love to hear thoughts and feedback.**

TorchArrow is a [torch](https://github.com/pytorch/pytorch).Tensor-like Python DataFrame library for data preprocessing in PyTorch models, with two high-level features:

* DataFrame library (like Pandas) with strong GPU or other hardware acceleration (under development) and PyTorch ecosystem integration.
* Columnar memory layout based on [Apache Arrow](https://arrow.apache.org/docs/format/Columnar.html#physical-memory-layout) with strong variable-width and nested data support (such as string, list, map) and Arrow ecosystem integration.

## Installation

You will need Python 3.7 or later. Also, we highly recommend installing an [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) environment.

First, set up an environment. If you are using conda, create a conda environment:
```
conda create --name torcharrow python=3.7
conda activate torcharrow
```

### Version Compatibility

The following is the corresponding `torcharrow` versions and supported Python versions.

| `torch`            | `torcharrow`        | `python`          |
| ------------------ | ------------------ | ----------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.7`, `<=3.10` |
| `1.13.0`           | `0.2.0`            | `>=3.7`, `<=3.10` |


### Colab

Follow the instructions [in this Colab notebook](https://colab.research.google.com/drive/1S0ldwN7qNM37E4WZnnAEnzn1DWnAQ6Vt)

### Nightly Binaries

Experimental nightly binary on macOS (requires macOS SDK >= 10.15) and Linux (requires glibc >= 2.17) for Python 3.7, 3.8, and 3.9 can be installed via pip wheels:
```
pip install --pre torcharrow -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

### From Source

If you are installing from source, you will need Python 3.7 or later and a C++17 compiler.

#### Get the TorchArrow Source
```bash
git clone --recursive https://github.com/pytorch/torcharrow
cd torcharrow
# if you are updating an existing checkout
git submodule sync --recursive
git submodule update --init --recursive
```

#### Install Dependencies

On macOS

[HomeBrew](https://brew.sh/) is required to install development tools on macOS.

```bash
# Install dependencies from Brew
brew install --formula ninja flex bison cmake ccache icu4c boost gflags glog libevent

# Build and install other dependencies
scripts/build_mac_dep.sh ranges_v3 fmt double_conversion folly re2
```

On Ubuntu (20.04 or later)
```bash
# Install dependencies from APT
apt install -y g++ cmake ccache ninja-build checkinstall \
    libssl-dev libboost-all-dev libdouble-conversion-dev libgoogle-glog-dev \
    libgflags-dev libevent-dev libre2-dev libfl-dev libbison-dev
# Build and install folly and fmt
scripts/setup-ubuntu.sh
```

#### Install TorchArrow
For local development, you can build with debug mode:
```
DEBUG=1 python setup.py develop
```

And run unit tests with
```
python -m unittest -v
```

To build and install TorchArrow with release mode:
```
python setup.py install
```


## Documentation
You can find the API documentation [here](https://pytorch.org/torcharrow/).

This [10 minutes tutorial](https://github.com/pytorch/torcharrow/blob/main/tutorial/tutorial.ipynb) provides a short introduction to TorchArrow, and you can also try it in [this Colab](https://colab.research.google.com/drive/1mQ3S6dwmU-zhBe2Tdvq_VRAnjQ3paiay).

## Examples
You can find the example about integrating a TorchRec based training loop utilizing TorchArrow's on-the-fly preprocessing
[here](https://github.com/pytorch/torchrec/tree/main/examples/torcharrow). More examples are coming soon!

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## Beta Usage and Feedback

We'd love to hear from and work with early adopters to shape our design. Please reach out by raising an issue if you're
interested in using this library for your project.

## Future Plans
We hope to continue to expand the library, harden API, and gather feedback to enable future releases. Stay tuned!

## License

TorchArrow is BSD licensed, as found in the [LICENSE](LICENSE) file.
