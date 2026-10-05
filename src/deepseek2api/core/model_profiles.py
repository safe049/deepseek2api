from __future__ import annotations

from dataclasses import dataclass

from ..config import EXPOSED_MODELS


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    native_function_calling: bool
    preferred_format: str
    stream_handler_type: str


def get_all_models() -> list[str]:
    return list(EXPOSED_MODELS)


def is_valid_model(model: str) -> bool:
    return model in EXPOSED_MODELS