import json
import pathlib
import typing

import pydantic


class QuestionnaireItem(pydantic.BaseModel):
    id: int
    content: str
    dimension: str


class QuestionnaireSegment(pydantic.BaseModel):
    label: str
    task: str
    questions: typing.List[QuestionnaireItem]
    scale: typing.Dict[int | str, str]

    def get_question(self, index: int) -> QuestionnaireItem:
        return next(filter(lambda question: question.id == index, self.questions))

    def __len__(self):
        return len(self.questions)


class Questionnaire(pydantic.BaseModel):
    path: pathlib.Path

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def segments(self) -> typing.List[QuestionnaireSegment]:
        return [
            QuestionnaireSegment.model_validate(segment)
            for segment in json.load(open(self.path))
        ]

    def get_question(self, segment: str, index: int) -> QuestionnaireItem:
        return next(filter(lambda seg: seg.label == segment, self.segments)).get_question(index)

    def __len__(self) -> int:
        return sum([len(segment) for segment in self.segments])


__all__ = ["QuestionnaireItem", "QuestionnaireSegment", "Questionnaire"]
