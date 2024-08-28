import json
import pathlib
import typing

import pydantic

from llm_moral_values.questionnaire import schemas


class Survey(pydantic.BaseModel):
    path: pathlib.Path

    questionnaire_file: str = "questionnaire.json"
    collections_file: str = "collections.json"

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def segments(self) -> typing.List[schemas.Segment]:
        return [
            schemas.Segment.model_validate(segment) for segment in json.load(open(self.path / self.questionnaire_file))
        ]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def collections(self) -> typing.List[schemas.Collection]:
        return [
            schemas.Collection(label=label, groups=groups)
            for label, groups in json.load(open(self.path / self.collections_file)).items()
        ]

    def get_question(self, segment: str, id: int) -> schemas.Question:
        return next(filter(lambda seg: seg.label == segment, self.segments)).get_question(id)

    def get_collection(self, label: str) -> schemas.Collection:
        return next(filter(lambda collection: collection.label == label, self.collections))

    def __len__(self) -> int:
        return sum([len(segment) for segment in self.segments])


__all__ = ["Survey", "schemas"]
