from __future__ import annotations

import base64
import bisect
import datetime
import json
import uuid
import warnings
from collections import defaultdict
from functools import reduce, singledispatch, wraps
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, TypeVar

import numpy as np
import torch
from pyannote.core import Segment as PC_Segment
from pydantic import BaseModel, field_serializer, field_validator


def time_formatter(value: float) -> str:
    td = datetime.timedelta(seconds=value)
    dt = datetime.datetime.min + td
    return dt.strftime("%H:%M:%S,%f")[:-3]


# parse string to seconds
def time_parser(value: str) -> float:
    dt = datetime.datetime.strptime(value, "%H:%M:%S,%f")
    return dt.hour * 60 * 60 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


class IdentifiedSegment(BaseModel, PC_Segment):
    """An abstract class for segments with ids. Has the properties:

    - id: str
    - start: float
    - end: float
    """

    id: Optional[str] = None
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @duration.setter
    def duration(self, value: float) -> None:
        self.end = self.start + value

    @field_validator("start", "end", mode="before")
    @classmethod
    def validate_time(cls, v):
        if type(v) == str:
            return time_parser(v)
        return v

    @field_serializer("start", "end")
    def serialize_time(v: float) -> str:
        return time_formatter(v)

class Word(IdentifiedSegment):
    text: str


if __name__=="__main__":
    pass