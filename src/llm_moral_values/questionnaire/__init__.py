import json
import pathlib
import typing

import pydantic

from llm_moral_values.questionnaire import schemas


class Questionnaire(pydantic.BaseModel):
    path: pathlib.Path

    questionnaire_file: str = "questionnaire.json"
    surveys_file: str = "surveys.json"

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def segments(self) -> typing.List[schemas.Segment]:
        return [
            schemas.Segment.model_validate(segment) for segment in json.load(open(self.path / self.questionnaire_file))
        ]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def surveys(self) -> typing.List[schemas.Survey]:
        return [
            schemas.Survey(label=label, groups=groups)
            for label, groups in json.load(open(self.path / self.surveys_file)).items()
        ]

    def get_question(self, segment: str, id: int) -> schemas.Question:
        return next(filter(lambda seg: seg.label == segment, self.segments)).get_question(id)

    def get_survey(self, label: str) -> schemas.Survey:
        return next(filter(lambda survey: survey.label == label, self.surveys))

    def __len__(self) -> int:
        return sum([len(segment) for segment in self.segments])


__all__ = ["Questionnaire", "schmemas"]
