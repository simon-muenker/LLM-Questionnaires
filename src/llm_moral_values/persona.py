import json
import pathlib

import pydantic


class Persona(pydantic.BaseModel):
    id: str
    content: str | None = None

    @classmethod
    def from_json(cls, path: pathlib.Path) -> "Persona":
        return cls(**json.load(open(path)))
