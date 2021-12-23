# TorchArrow (Warning: Unstable Prototype)

**This is a prototype library currently under heavy development. It does not currently have stable releases, and as such will likely be modified significantly in backwards compatibility breaking ways until alpha release (targeting early 2022). If you have suggestions on the API or use cases you would like to be covered, please open a GitHub issue. We would love to hear thoughts and feedback.**

TorchArrow is a [torch](https://github.com/pytorch/pytorch).Tensor-like Python DataFrame library for data preprocessing in deep learning. It supports multiple execution runtimes and [Arrow](https://github.com/apache/arrow) as a common format.

It plans to provide:

* Python Dataframe library focusing on streaming-friendly APIs for data preprocessing in deep learning
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

### Colab (Experimental)

Follow the instructions [in this Colab notebook](https://colab.research.google.com/drive/1S0ldwN7qNM37E4WZnnAEnzn1DWnAQ6Vt)

### Binaries (Experimental)

Experimental binary on MacOS and Linux (Ubuntu 18.04 or later, CentOS 8 or later) can be installed via pip wheels:
```
pip install torcharrow
```

Make sure you have `pip>=20.3` on Linux to install the wheel.

### From Source

If you are installing from source, you will need Python 3.7 or later and a C++17 compiler.

#### Get the TorchArrow Source
```bash
git clone --recursive https://github.com/facebookresearch/torcharrow
cd torcharrow
# if you are updating an existing checkout
git submodule sync --recursive
git submodule update --init --recursive
```

#### Install Dependencies

On MacOS

[HomeBrew](https://brew.sh/) is required to install development tools on MacOS.

```bash
# Install dependencies from Brew
brew install --formula ninja cmake ccache protobuf icu4c boost gflags glog libevent lz4 lzo snappy xz zstd

# Build and install other dependencies
scripts/build_mac_dep.sh ranges_v3 googletest fmt double_conversion folly re2
```

On Ubuntu (20.04 or later)
```bash
# Install dependencies from APT
apt install -y g++ cmake ccache ninja-build checkinstall \
    libssl-dev libboost-all-dev libdouble-conversion-dev libgoogle-glog-dev \
    libbz2-dev libgflags-dev libgtest-dev libgmock-dev libevent-dev libfmt-dev \
    libprotobuf-dev liblz4-dev libzstd-dev libre2-dev libsnappy-dev liblzo2-dev \
    protobuf-compiler
# Build and install Folly
scripts/install_ubuntu_folly.sh
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
This [10 minutes tutorial](https://github.com/facebookresearch/torcharrow/blob/main/tutorial/tutorial.ipynb) provides a short introduction to TorchArrow, and you can also try it in [this Colab](https://colab.research.google.com/drive/1mQ3S6dwmU-zhBe2Tdvq_VRAnjQ3paiay). More documents on advanced topics are coming soon!

## Future Plans
We hope to sufficiently expand the library, harden APIs, and gather feedback to enable a alpha release (early 2022).

## License

TorchArrow is BSD licensed, as found in the [LICENSE](LICENSE) file.
