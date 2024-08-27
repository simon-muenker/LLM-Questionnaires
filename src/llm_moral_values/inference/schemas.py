import typing
import pydantic


type Roles = typing.Literal["user", "assistant", "system"]

type Models = typing.Literal[
    # gemma family (Google)
    "gemma:7b-instruct-q6_K",
    "gemma2:27b-instruct-q6_K",
    # llama family (MetaAI)
    "llama3.1:8b-instruct-q6_K",
    "llama3.1:70b-instruct-q6_K",
    # mi(s/x)tral family (Mistral AI)
    "mistral:7b-instruct-v0.3-q6_K",
    # disabled due to size issues
    # "mistral-large:123b-instruct-2407-q6_K",
    "mixtral:8x7b-instruct-v0.1-q6_K",
    "mixtral:8x22b-instruct-v0.1-q6_K",
    # alibaba
    "qwen2:72b-instruct-q6_K",
    # mircosoft
    "phi3:14b-medium-128k-instruct-q6_K"
]


class Message(pydantic.BaseModel):
    role: Roles
    content: str


class Chat(pydantic.BaseModel):
    messages: typing.List[Message]

    def __getitem__(self, index: int):
        return self.messages[index]

    def __iter__(self):
        return iter(self.messages)

    def add_message(self, message: Message) -> "Chat":
        return Chat(messages=[*self, message])

    def remove_message(self, index: int) -> "Chat":
        return Chat(messages=[self[:index] + self[index + 1 :]])

    def to_json(self, path: str) -> None:
        open(path, "w").write(self.model_dump_json(indent=4))

    @pydantic.field_validator("messages")
    @classmethod
    def check_messages_base_integrity(
        cls, messages: typing.List[Message]
    ) -> typing.List[Message]:
        if len(messages) < 2:
            raise ValueError("The chat must contain at least two messages.")

        if messages[0].role != "system":
            raise ValueError("The first message must be the system message.")

        if messages[1].role != "user":
            raise ValueError("The second message must be a user message.")

        return messages
