# Building torcharrow packages for release

## Anaconda packages
Coming soon!

## Wheels

### Linux
Build the wheel:
```
PYTHON_VERSION=3.7 packaging/build_wheel.sh
```

Fix the wheel with `auditwheel`:
```
# Use the envirnoment that builds the wheel
conda activate env3.7

pip install auditwheel
auditwheel repair dist/*.whl -w fixed_dist --plat manylinux_2_27_x86_64
```

To upload wheels,

```
pip install twine
twine upload fixed_dist/*.whl
```

### OSX
Build the wheel:
```
PYTHON_VERSION=3.7 packaging/build_wheel.sh
```

Fix the wheel with `delocate`:
```
# Use the envirnoment that builds the wheel
conda activate env3.7

pip install delocate
delocate-wheel -w fixed_dist -v dist/*.whl
```

To upload wheels,

```
pip install twine
twine upload fixed_dist/*.whl
```
