import json
import pathlib
import typing

import pydantic

import src


class Questionnaire(pydantic.BaseModel):
    path: pathlib.Path

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def questions(self) -> typing.Dict:
        return json.load(open(self.path / "questions.json"))

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def mapping(self) -> typing.Dict:
        return json.load(open(self.path / "mapping.json"))

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def survey(self) -> typing.Dict:
        return json.load(open(self.path / "survey.json"))

    def answer_questionnaire(
            self,
            model: str,
            prefix: str | None = None
    ) -> typing.Iterator[typing.Dict]:

        for label, segment in self.questions.items():
            for n, question in segment["questions"].items():
                observation: typing.Dict = {
                    "category": label,
                    "number": n,
                    "model": model,
                    "response": None
                }

                system_prompt: str = f"{prefix}\n\n{segment['task']}" if prefix else segment['task']

                observation["response"] = (
                    src.api.Inference(
                        model=model,
                        system_prompt=system_prompt
                    )
                    (f"{system_prompt}\n\n{question}")
                    .extract_numeric_answer()
                )

                yield observation

    def __len__(self) -> int:
        return len([
            n
            for segment in self.questions.values()
            for n in segment["questions"].keys()
        ])
