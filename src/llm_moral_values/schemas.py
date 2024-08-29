import pathlib
import typing

import pydantic

from llm_moral_values.inference.schemas import Models


class Persona(pydantic.BaseModel):
    id: str
    content: str | None = None

    @classmethod
    def from_directory(cls, source_path: pathlib.Path) -> typing.List["Persona"]:
        return [cls.model_validate_json(open(path).read()) for path in list(source_path.iterdir())]


class Model(pydantic.BaseModel):
    id: Models

    @classmethod
    def from_inference_selection(cls) -> typing.List["Model"]:
        return [cls(id=model) for model in typing.get_args(Models)]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def name(self) -> str:
        return self.id.split("-")[0]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def dir_name(self) -> str:
        return self.name.replace(":", "-")


__all__ = ["Persona", "Model"]
