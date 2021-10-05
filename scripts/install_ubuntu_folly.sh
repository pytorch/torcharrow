# Based on https://github.com/facebookincubator/velox/blob/475bb6538c3e57c8c347bca76cb783e315c9c261/scripts/setup-ubuntu.sh#L55-L80

# Folly must be built with the same compiler flags so that some low level types
# are the same size.
export COMPILER_FLAGS="-mavx2 -mfma -mavx -mf16c -masm=intel -mlzcnt"
BUILD_DIR=_build

# Enter build directory.
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

function install_folly {
  local NAME="folly"

  if [ -d "${NAME}" ]; then
    read -p "Do you want to rebuild '${NAME}'? (y/N) " confirm
    if [[ "${confirm}" =~ ^[Yy]$ ]]; then
      rm -rf "${NAME}"
    else
      return 0
    fi
  fi

  git clone https://github.com/facebook/folly.git "${NAME}"
  cd "${NAME}"
  cmake \
    -DCMAKE_CXX_FLAGS="$COMPILER_FLAGS" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -GNinja \
    -DFOLLY_HAVE_INT128_T=1 \
    .
  ninja
  sudo checkinstall -y ninja install
}

install_folly
