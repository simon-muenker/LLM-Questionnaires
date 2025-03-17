import json
import pathlib
import typing

import pydantic


class SurveySegment(pydantic.BaseModel):
    label: str
    groups: typing.Dict[str, typing.Dict[str, typing.Dict[str, float]]]


class Survey(pydantic.BaseModel):
    path: pathlib.Path

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def segments(self) -> typing.List[SurveySegment]:
        return [
            SurveySegment(label=label, groups=groups)
            for label, groups in json.load(open(self.path)).items()
        ]

    def get_survey(self, label: str) -> SurveySegment:
        return next(filter(lambda collection: collection.label == label, self.items))

    def __len__(self) -> int:
        return len(self.segments)


__all__ = ["SurveySegment", "Survey"]
