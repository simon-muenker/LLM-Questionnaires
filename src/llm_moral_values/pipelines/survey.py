import glob
import itertools
import json
import logging
import pathlib
import re
import typing
import uuid

import pydantic
import tqdm

import cltrier_lib

from llm_moral_values import data, questionnaire, schemas


class ConductSurvey(pydantic.BaseModel):
    iterations: int
    models: typing.List[schemas.Model]

    survey: questionnaire.Survey
    personas: typing.List[schemas.Persona]

    export_path: pathlib.Path

    def __call__(self):
        logging.info("> Conducting Survey")
        for model, persona in list(itertools.product(self.models, self.personas)):
            self.process_configuration(model, persona)

        logging.info("> Collate Data")
        data_survey: data.Survey = data.Survey.from_samples(f"{self.export_path}/**/*.json")
        data_survey.data.to_parquet(f"{self.export_path}/survey.parquet")

        logging.info("> Write Data Report")
        data_survey.write_report(f"{self.export_path}/survey.report.txt")

        logging.info("> Generate Cross Evaluation")
        data_cross_evaluation: data.CrossEvaluation = data.CrossEvaluation.from_survey(
            data_survey, self.survey.get_collection("graham_et_al")
        )
        data_cross_evaluation.data.to_parquet(f"{self.export_path}/cross_evaluation.parquet")

    def process_configuration(self, model: schemas.Model, persona: schemas.Persona) -> None:
        iteration_path: pathlib.Path = self.export_path / persona.id / model.dir_name
        iteration_path.mkdir(parents=True, exist_ok=True)

        while len(glob.glob(f"{iteration_path}/*.json")) < self.iterations:
            self.process_answers(model, persona, iteration_path)
        logging.info(f"Generated {self.iterations} surveys for configuration: {model.name}:{persona.id}")

    def process_answers(self, model: schemas.Model, persona: schemas.Persona, export_path: pathlib.Path) -> None:
        json.dump(
            [
                item
                for item in tqdm.tqdm(
                    ConductSurvey.answer_survey(self.survey, model, persona),
                    total=len(self.survey),
                    desc=f"{(model.name, persona.id)}",
                )
            ],
            open(export_path / f"{uuid.uuid4()}.json", "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )

    @staticmethod
    def answer_survey(
        survey: questionnaire.Survey,
        model: schemas.Model,
        persona: schemas.Persona,
    ) -> typing.Iterator[typing.Dict]:
        for segment in survey.segments:
            for question in segment.questions:
                response: cltrier_lib.inference.schemas.Chat = cltrier_lib.inference.Pipeline(model=model.id)(
                    ConductSurvey.prepare_chat(persona, segment, question)
                )

                yield {
                    "segment": segment.label,
                    "id": question.id,
                    "dimension": question.dimension,
                    "model": model.id,
                    "persona": persona.id,
                    "response": ConductSurvey.extract_numeric_answer(response),
                }

    @staticmethod
    def prepare_chat(
        persona: schemas.Persona, segment: questionnaire.schemas.Segment, question: questionnaire.schemas.Question
    ) -> cltrier_lib.inference.schemas.Chat:
        return cltrier_lib.inference.schemas.Chat(
            messages=[
                cltrier_lib.inference.schemas.Message(role="system", content=str(persona.content)),
                cltrier_lib.inference.schemas.Message(role="user", content=f"{segment.task}\n\nSentence: {question.content}"),
            ]
        )

    @staticmethod
    def extract_numeric_answer(response: cltrier_lib.inference.schemas.Chat) -> int | None:
        extracted_response: typing.Match | None = re.search(r"(\d)", response[-1].content)

        return extracted_response.group(1) if extracted_response else None
