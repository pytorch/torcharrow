# TorchArrow

TorchArrow is a [Pandas](https://github.com/pandas-dev/pandas) inspired DataFrame library in Python built on the [Apache Arrow](https://github.com/apache/arrow) columnar memory format and
leveraging the [Velox vectorized engine](https://github.com/facebookexternal/velox/) for loading, filtering, mapping, joining, aggregating, and
otherwise manipulating tabular data on CPUs.

TorchArrow supports [PyTorch](https://github.com/pytorch/pytorch)'s Tensors as first class citizens. It allows mostly zero copy interop with Numpy, Pandas, PyArrow, CuDf and of course integrates well with PyTorch data-wrangling workflows.


## Installation

### Binaries

Coming soon!

### From Source

If you are installing from source, you will need Python 3.8 or later and a C++17 compiler. Also, we highly recommend installing an [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) environment.

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

To install TorchArrow with release mode (WARNING: may take very long to build):
```
python setup.py install
```


## Documentation
All documentation is available via Notebooks:
* [TorchArrow in 10 minutes (a tutorial)](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow10min.ipynb)
* [TorchArrow data pipes](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow_data_pipes.ipynb)
* [TorchArrow state handling](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow_state.ipynb)
* TorchArrow multitargeting - TBD
* TorchArrow, Pandas, UPM and SQL: What's the difference - TBD
* TorchArrow, Design rationale - TBD

## Status
This directory supports rapid development. So expect frequent changes.

Still to be done:
* Add tabular.py as package in setup and not as code
* [DONE] How to do Multi-device targeting (See TorchArrow state handling notebook)
* An example program analysis (types/PPF?)
* Add example UDFs
* Add Tensors as example UDTs
* [WORKS, example to be written] Using Numba for Jitting

## License

TorchArrow is BSD licensed, as found in the [LICENSE](LICENSE) file.

