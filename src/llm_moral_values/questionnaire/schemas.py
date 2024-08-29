import typing

import pydantic


class Question(pydantic.BaseModel):
    id: int
    content: str
    dimension: str


class Segment(pydantic.BaseModel):
    label: str
    task: str
    questions: typing.List[Question]
    scale: typing.Dict[int | str, str]

    def get_question(self, index: int) -> Question:
        return next(filter(lambda question: question.id == index, self.questions))

    def __len__(self):
        return len(self.questions)


class Collection(pydantic.BaseModel):
    label: str
    groups: typing.Dict[str, typing.Dict[str, typing.Dict[str, float]]]
