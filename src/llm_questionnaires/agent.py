import pathlib
import typing

import pydantic
from pydantic_ai import Agent as OpenAIAgent, ModelRequestNode
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class AgentPersona(pydantic.BaseModel):
    id: str
    content: str | None = None

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def dir_name(self) -> str:
        return self.id.lower().replace(" ", "-")

    @classmethod
    def from_json(cls, source_file: pathlib.Path) -> "AgentPersona":
        return cls.model_validate_json(open(source_file).read())

    @classmethod
    def from_directory(cls, source_path: pathlib.Path) -> typing.List["AgentPersona"]:
        return sorted(
            [cls.from_json(source_file) for source_file in list(source_path.iterdir())]
        )

    def __lt__(self, other: "AgentPersona") -> bool:
        return self.id < other.id


class AgentModel(pydantic.BaseModel):
    id: str

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def dir_name(self) -> str:
        return self.id.lower().replace(":", "-")


class Agent(pydantic.BaseModel):
    persona: AgentPersona
    model: AgentModel

    _endpoint: OpenAIModel
    _memory: typing.List[ModelRequestNode] = []
    _memory_length: int = 5

    def model_post_init(self, _: typing.Any):
        self._endpoint = OpenAIModel(
            model_name=self.model.id,
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )

    def __call__(self, user_prompt: str, result_type: typing.Any = str, use_memory: bool = False) -> AgentRunResult:
        try:
            result = (
                OpenAIAgent(
                    self._endpoint,
                    result_type=result_type,
                    system_prompt=self.persona.content,
                )
                .run_sync(user_prompt, **dict(message_history=self._memory if use_memory else None))
            )
            if use_memory:
                self._refresh_memory(result.new_messages())
            return result.data
            

        except UnexpectedModelBehavior:
            return None
        
    def _refresh_memory(self, node: ModelRequestNode) -> None:
        self._memory.extend(node)
        self._memory = self._memory[-(self._memory_length * 3):]



__all__ = ["AgentPersona", "AgentModel", "Agent"]
