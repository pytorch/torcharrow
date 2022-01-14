#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import distutils.command.clean
import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()


def _get_version():
    version = "0.1.0a0"
    sha = "Unknown"
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR))
            .decode("ascii")
            .strip()
        )
    except Exception:
        pass

    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "torcharrow" / "version.py"
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


VERSION, SHA = _get_version()
_export_version(VERSION, SHA)

print("-- Building version " + VERSION)


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torcharrow extension
        for path in (ROOT_DIR / "torcharrow").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


# Based off of
# https://github.com/pytorch/audio/blob/2c8aad97fc8d7647ee8b2df2de9312cce0355ef6/build_tools/setup_helpers/extension.py#L46
class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.")
        super().run()

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        if "DEBUG" in os.environ:
            # TODO: is there any better approach?
            cfg = "Debug" if os.environ["DEBUG"] == "1" else "Release"
        else:
            cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            # f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            # "-DCMAKE_VERBOSE_MAKEFILE=ON",
            # f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
            "-DVELOX_CODEGEN_SUPPORT=OFF",
            "-DVELOX_BUILD_MINIMAL=ON",
        ]
        build_args = ["--target", "install"]

        # Default to Ninja
        if "CMAKE_GENERATOR" not in os.environ:
            cmake_args += ["-GNinja"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", str(ROOT_DIR)] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="torcharrow",
    version=VERSION,
    description="A Pandas inspired, Arrow compatible, Velox supported dataframe library for PyTorch",
    url="https://github.com/pytorch/torcharrow",
    author="Facebook",
    author_email="packages@pytorch.org",
    license="BSD",
    install_requires=[
        "arrow",
        "cffi",
        "numpy==1.21.4",
        "pandas<=1.3.5", # Last version that has Python 3.7 wheel
        "typing",
        "tabulate",
        "typing-inspect",
        "pyarrow",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages()  # TODO ???
    + find_packages(where="./velox_rt")
    + find_packages(where="./numpy_rt")
    + find_packages(where="./test"),
    zip_safe=False,
    ext_modules=[Extension(name="torcharrow._torcharrow", sources=[])],
    cmdclass={
        "build_ext": CMakeBuild,
        "clean": clean,
    },
)
