# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import re
import typing as ty
from abc import ABC, abstractmethod
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Aux

# Pretty printing constants; reused everywhere
# TODO: Inline these constants.
OPEN = "{"
CLOSE = "}"
NL = "\n"


# -----------------------------------------------------------------------------
# Schema and Field

MetaData = ty.Dict[str, str]


@dataclass(frozen=True)
class Field:
    name: str
    dtype: "DType"
    metadata: ty.Optional[MetaData] = None

    def __str__(self):
        meta = ""
        if self.metadata is not None:
            meta = (
                f"meta = {OPEN}{', '.join(f'{k}: {v}' for k,v in self.metadata)}{CLOSE}"
            )
        return f"Field('{self.name}', {str(self.dtype)}{meta})"


# -----------------------------------------------------------------------------
# Immutable Types with structural equality...


@dataclass(frozen=True)  # type: ignore
class DType(ABC):
    typecode: ty.ClassVar[str] = "__TO_BE_DEFINED_IN_SUBCLASS__"
    arraycode: ty.ClassVar[str] = "__TO_BE_DEFINED_IN_SUBCLASS__"

    @property
    @abstractmethod
    def nullable(self):
        return False

    @property
    def py_type(self):
        return type(self.default_value())

    def __str__(self):
        if self.nullable:
            return f"{self.name.title()}(nullable=True)"
        else:
            return self.name

    @abstractmethod
    def constructor(self, nullable):
        pass

    def with_null(self, nullable=True):
        return self.constructor(nullable)

    def default_value(self):
        # must be overridden by all non primitive types!
        return type(self).default


# for now: no float16, and all date and time stuff, no categorical, (and Null is called Void)


@dataclass(frozen=True)  # type: ignore
class Numeric(DType):
    pass


@dataclass(frozen=True)
class Boolean(DType):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "b"
    arraycode: ty.ClassVar[str] = "b"
    name: ty.ClassVar[str] = "boolean"
    default: ty.ClassVar[bool] = False

    def constructor(self, nullable):
        return Boolean(nullable)


@dataclass(frozen=True)
class Int8(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "c"
    arraycode: ty.ClassVar[str] = "b"
    name: ty.ClassVar[str] = "int8"
    default: ty.ClassVar[int] = 0

    def constructor(self, nullable):
        return Int8(nullable)


@dataclass(frozen=True)
class Int16(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "s"
    arraycode: ty.ClassVar[str] = "h"
    name: ty.ClassVar[str] = "int16"
    default: ty.ClassVar[int] = 0

    def constructor(self, nullable):
        return Int16(nullable)


@dataclass(frozen=True)
class Int32(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "i"
    arraycode: ty.ClassVar[str] = "i"
    name: ty.ClassVar[str] = "int32"
    default: ty.ClassVar[int] = 0

    def constructor(self, nullable):
        return Int32(nullable)


@dataclass(frozen=True)
class Int64(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "l"
    arraycode: ty.ClassVar[str] = "l"
    name: ty.ClassVar[str] = "int64"
    default: ty.ClassVar[int] = 0

    def constructor(self, nullable):
        return Int64(nullable)


# Not all Arrow types are supported. We don't have a backend to support unsigned
# integer types right now so they are removed to not confuse users. Feel free to
# add unsigned int types when we have a supporting backend.


@dataclass(frozen=True)
class Float32(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "f"
    arraycode: ty.ClassVar[str] = "f"
    name: ty.ClassVar[str] = "float32"
    default: ty.ClassVar[float] = 0.0

    def constructor(self, nullable):
        return Float32(nullable)


@dataclass(frozen=True)
class Float64(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "g"
    arraycode: ty.ClassVar[str] = "d"
    name: ty.ClassVar[str] = "float64"
    default: ty.ClassVar[float] = 0.0

    def constructor(self, nullable):
        return Float64(nullable)


@dataclass(frozen=True)
class String(DType):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "u"  # utf8 string (n byte)
    arraycode: ty.ClassVar[str] = "w"  # wchar_t (2 byte)
    name: ty.ClassVar[str] = "string"
    default: ty.ClassVar[str] = ""

    def constructor(self, nullable):
        return String(nullable)


@dataclass(frozen=True)
class Map(DType):
    key_dtype: DType
    item_dtype: DType
    nullable: bool = False
    keys_sorted: bool = False
    name: ty.ClassVar[str] = "Map"
    typecode: ty.ClassVar[str] = "+m"
    arraycode: ty.ClassVar[str] = ""

    @property
    def py_type(self):
        return ty.Dict[self.key_dtype.py_type, self.item_dtype.py_type]

    def constructor(self, nullable):
        return Map(self.key_dtype, self.item_dtype, nullable)

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        return f"Map({self.key_dtype}, {self.item_dtype}{nullable})"

    def default_value(self):
        return {}


@dataclass(frozen=True)
class List(DType):
    item_dtype: DType
    nullable: bool = False
    fixed_size: int = -1
    name: ty.ClassVar[str] = "List"
    typecode: ty.ClassVar[str] = "+l"
    arraycode: ty.ClassVar[str] = ""

    @property
    def py_type(self):
        return ty.List[self.item_dtype.py_type]

    def constructor(self, nullable, fixed_size=-1):
        return List(self.item_dtype, nullable, fixed_size)

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        fixed_size = (
            ", fixed_size=" + str(self.fixed_size) if self.fixed_size >= 0 else ""
        )
        return f"List({self.item_dtype}{nullable}{fixed_size})"

    def default_value(self):
        return []


@dataclass(frozen=True)
class Struct(DType):
    fields: ty.List[Field]
    nullable: bool = False
    is_dataframe: bool = False
    metadata: ty.Optional[MetaData] = None
    name: ty.ClassVar[str] = "Struct"
    typecode: ty.ClassVar[str] = "+s"
    arraycode: ty.ClassVar[str] = ""

    # For generating NamedTuple class name for cached _py_type (done in __post__init__)
    _global_py_type_id: ty.ClassVar[int] = 0
    _local_py_type_id: int = dataclasses.field(compare=False, default=-1)

    # TODO: Use utility method (instead of built-in method in DType for such ops)
    def get_index(self, name: str) -> int:
        for idx, field in enumerate(self.fields):
            if field.name == name:
                return idx
        # pyre-fixme[7]: Expected `int` but got `None`.
        return None

    def __getstate__(self):
        # _py_type is NamedTuple which is not pickle-able, skip it
        return (self.fields, self.nullable, self.is_dataframe, self.metadata)

    def __setstate__(self, state):
        # Restore state, __setattr__ hack is needed due to the frozen dataclass
        object.__setattr__(self, "fields", state[0])
        object.__setattr__(self, "nullable", state[1])
        object.__setattr__(self, "is_dataframe", state[2])
        object.__setattr__(self, "metadata", state[3])

        # reconstruct _py_type
        self.__post_init__()

    def __post_init__(self):
        if self.nullable:
            for f in self.fields:
                if not f.dtype.nullable:
                    raise TypeError(
                        f"nullable structs require each field (like {f.name}) to be nullable as well."
                    )
        object.__setattr__(self, "_local_py_type_id", type(self)._global_py_type_id)
        type(self)._global_py_type_id += 1

    def _set_py_type(self):
        # cache the type instance, __setattr__ hack is needed due to the frozen dataclass
        # the _py_type is not listed above to avoid participation in equality check

        def fix_name(name, idx):
            # Anonomous Row
            if name == "":
                return "f_" + str(idx)

            # Remove invalid character for NamedTuple
            # TODO: this might cause name duplicates, do disambiguation
            name = re.sub("[^a-zA-Z0-9_]", "_", name)
            if name == "" or name[0].isdigit() or name[0] == "_":
                name = "f_" + name
            return name

        object.__setattr__(
            self,
            "_py_type",
            ty.NamedTuple(
                "TorchArrowGeneratedStruct_" + str(self._local_py_type_id),
                [
                    (fix_name(f.name, idx), f.dtype.py_type)
                    for (idx, f) in enumerate(self.fields)
                ],
            ),
        )

    @property
    def py_type(self):
        if not hasattr(self, "_py_type"):
            # this call is expensive due to the namedtuple creation, so
            # do it lazily
            self._set_py_type()
        return self._py_type

    def constructor(self, nullable):
        return Struct(self.fields, nullable)

    def get(self, name):
        for f in self.fields:
            if f.name == name:
                return f.dtype
        raise KeyError(f"{name} not among fields")

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        fields = f"[{', '.join(str(f) for f in self.fields)}]"
        meta = ""
        if self.metadata is not None:
            meta = f", meta = {OPEN}{', '.join(f'{k}: {v}' for k,v in self.metadata)}{CLOSE}"
        else:
            return f"Struct({fields}{nullable}{meta})"

    def default_value(self):
        return tuple(f.dtype.default_value() for f in self.fields)
