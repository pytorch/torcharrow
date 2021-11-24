# Building torcharrow packages for release

## Anaconda packages
Coming soon!

## Wheels

### Linux
Coming soon!

### OSX
Build the wheel:
```
PYTHON_VERSION=3.8 packaging/build_wheel.sh
```

Fix the wheel with `delocate`:
```
pip install delocate
delocate-wheel -w fixed_dist -v dist/*.whl
```

To upload wheels,

```
pip install twine
twine upload fixed_dist/*.whl
```
