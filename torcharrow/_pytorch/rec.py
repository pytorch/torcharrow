import torch
import torcharrow._torcharrow as pyvelox
import torcharrow.dtypes as dt

from .common import ITorchConversion
from .common import _dtype_to_pytorch_dtype


class Dense(ITorchConversion):
    def __init__(self, *, batch_first=False, with_presence=False):
        self.batch_first = batch_first
        self.with_presence = with_presence

    def to_tensor(self, df):
        # Only support CPU dataframe for now
        assert df.device == "cpu"

        if self.with_presence:
            raise NotImplementedError

        if (not dt.is_struct(df.dtype)) or len(df.dtype.fields) == 0:
            raise ValueError(f"Unsupported dtype: {df.dtype}")

        data_tensor = torch.empty(
            (len(df.columns), len(df)),
            dtype=_dtype_to_pytorch_dtype(df.dtype.fields[0].dtype),
        )
        pyvelox._populate_dense_features_nopresence(df._data, data_tensor.data_ptr())

        if self.batch_first:
            # TODO: this would incur an extra copy. We can avoid this copy
            #    by fusing "tranpose-and-copy" when converting Velox vector to tensor.
            data_tensor = data_tensor.transpose(0, 1).contiguous()

        return data_tensor
