import pathlib
import typing

import pydantic


class AgentPersona(pydantic.BaseModel):
    id: str
    content: str | None = None

    @classmethod
    def from_directory(cls, source_path: pathlib.Path) -> typing.List["AgentPersona"]:
        return [cls.model_validate_json(open(path).read()) for path in list(source_path.iterdir())]


class AgentModel(pydantic.BaseModel):
    id: str

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def name(self) -> str:
        return self.id.split("-")[0]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def dir_name(self) -> str:
        return self.name.replace(":", "-")


class Agent(pydantic.BaseModel):
    pass


__all__ = ["AgentPersona", "AgentModel", "Agent"]
