.. TorchArrow documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TorchArrow Documentation
######################################
This library is part of the `PyTorch
<http://pytorch.org/>`_ project. PyTorch is an open source
deep learning framework.

TorchArrow is a torch.Tensor-like Python DataFrame library for data preprocessing in deep learning. 
It supports multiple execution runtimes and Arrow as a common format.

Features described in this documentation are classified by release status:

  *Stable:*  These features will be maintained long-term and there should generally
  be no major performance limitations or gaps in documentation.
  We also expect to maintain backwards compatibility (although
  breaking changes can happen and notice will be given one release ahead
  of time).

  *Beta:*  Features are tagged as Beta because the API may change based on
  user feedback, because the performance needs to improve, or because
  coverage across operators is not yet complete. For Beta features, we are
  committing to seeing the feature through to the Stable classification.
  We are not, however, committing to backwards compatibility.

  *Prototype:*  These features are typically not available as part of
  binary distributions like PyPI or Conda, except sometimes behind run-time
  flags, and are at an early stage for feedback and testing.


.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   torcharrow.rst
   column.rst
   dataframe.rst
   dtypes.rst

..
  TODO: docs about NumericColumn, StringColumn, DataFrame, dtype
  TODO: tutorial and demo

.. toctree::
   :maxdepth: 1
   :caption: PyTorch Libraries

   PyTorch <https://pytorch.org/docs>
   torchaudio <https://pytorch.org/audio>
   torchtext <https://pytorch.org/text>
   torchvision <https://pytorch.org/vision>
   torchdata <https://pytorch.org/data>
   TorchElastic <https://pytorch.org/elastic/>
   TorchServe <https://pytorch.org/serve>
   PyTorch on XLA Devices <http://pytorch.org/xla/>


Indices
==================

* :ref:`genindex`