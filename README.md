# TorchArrow

**This library is currently in the Beta stage and does not have a stable release. The API may change based on
user feedback or performance. We are committed to bring this library to stable release, but future changes may not be
completely backward compatible. If you have suggestions on the API or use cases you'd like to be covered, please open a
GitHub issue. We'd love to hear thoughts and feedback.**

TorchArrow is a machine learning preprocessing library over batch data, providing performant and Pandas-style easy-to-use API for model development. Currently it provides a Python DataFrame that allows extensible UDFs with [Velox](https://github.com/facebookincubator/velox/), with the following features:

* Seamless handoff with [PyTorch](https://github.com/pytorch/pytorch) or other model authoring, such as Tensor collation and easily plugging into PyTorch DataLoader and [DataPipes](https://github.com/pytorch/data#what-are-datapipes)
* Zero copy for external readers via [Arrow](https://github.com/apache/arrow) in-memory columnar format
* Multiple execution runtimes support:
    - High-performance CPU backend via [Velox](https://github.com/facebookincubator/velox/)
    - (Future Work) GPU backend via [libcudf](https://docs.rapids.ai/api/libcudf/stable/)
* High-performance C++ UDF support with vectorization

## Installation

You will need Python 3.7 or later. Also, we highly recommend installing an [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) environment.

First, set up an environment. If you are using conda, create a conda environment:
```
conda create --name torcharrow python=3.7
conda activate torcharrow
```

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
