import pydantic


class Persona(pydantic.BaseModel):
    id: str
    content: str | None = None


class Model(pydantic.BaseModel):
    id: str

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def name(self) -> str:
        return self.id.split("-")[0].replace(":", "-")


__all__ = ["Persona", "Model"]
