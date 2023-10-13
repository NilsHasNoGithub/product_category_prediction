import base64
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import orjson

T = TypeVar("T")
U = TypeVar("U")


def flatten(items: Iterable[Iterable[T]]) -> Iterable[T]:
    """Flatten 2d iterable into 1d"""
    return (x for l in items for x in l)


def split_into_n_chunks(items: Sequence[T], num_chunks: int) -> Iterable[List[T]]:
    """Split `items` into `num_chunks` chunks"""
    base_chunk_size, remainder = divmod(len(items), num_chunks)
    i_start = 0

    for i_chunk in range(num_chunks):
        chunk_size = base_chunk_size

        if remainder > i_chunk:
            chunk_size += 1

        yield items[i_start : i_start + chunk_size]

        i_start += chunk_size


def base64_encode(bts: bytes) -> str:
    """base64 encode bytes to string"""
    return base64.b64encode(bts).decode("utf-8")


def base64_decode(s: str) -> bytes:
    """base64 decode string to bytes"""
    return base64.b64decode(s)


def json_load(file_path: Union[str, Path]) -> Any:
    """Load json file into object"""
    with open(str(file_path), "rb") as f:
        return orjson.loads(f.read())


def map_opt(func: Callable[[T], U], item: Optional[T]) -> Optional[U]:
    if item is None:
        return None

    return func(item)
