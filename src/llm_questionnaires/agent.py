import pathlib
import typing

import pydantic
from pydantic_ai import Agent as OpenAIAgent
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class AgentPersona(pydantic.BaseModel):
    id: str
    content: str | None = None

    @classmethod
    def from_json(cls, source_file: pathlib.Path) -> "AgentPersona":
        return cls.model_validate_json(open(source_file).read())

    @classmethod
    def from_directory(cls, source_path: pathlib.Path) -> typing.List["AgentPersona"]:
        return [cls.from_json(source_file) for source_file in list(source_path.iterdir())]


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
    persona: AgentPersona
    model: AgentModel

    _endpoint: OpenAIModel

    def model_post_init(self, _: typing.Any):
        self._endpoint = OpenAIModel(
            model_name=self.model.id,
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )

    def __call__(self, user_prompt: str, result_type: typing.Any = str) -> AgentRunResult:
        try:
            return (
                OpenAIAgent(
                    self._endpoint,
                    result_type=result_type,
                    system_prompt=self.persona.content,
                )
                .run_sync(user_prompt)
                .data
            )

        except UnexpectedModelBehavior:
            return None


__all__ = ["AgentPersona", "AgentModel", "Agent"]
