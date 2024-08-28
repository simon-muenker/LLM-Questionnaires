import typing
import re
import itertools
import pathlib
import glob
import json
import uuid
import logging

import tqdm
import pydantic

from llm_moral_values import inference
from llm_moral_values import data
from llm_moral_values.questionnaire import Questionnaire
from llm_moral_values.persona import Persona


class ConductSurvey(pydantic.BaseModel):
    iterations: int
    models: typing.List[str]

    questionnaire: Questionnaire
    personas: typing.List[Persona]

    export_path: pathlib.Path

    def __call__(self):
        logging.info("> Conducting Survey")
        for model, persona in list(itertools.product(self.models, self.personas)):
            model_id: str = model.split("-")[0].replace(":", "-")

            prod_path: pathlib.Path = self.export_path / persona.id / model_id
            prod_path.mkdir(parents=True, exist_ok=True)

            while len(glob.glob(f"{prod_path}/*.json")) < self.iterations:
                json.dump(
                    [
                        item
                        for item in tqdm.tqdm(
                            ConductSurvey.answer_questionnaire(
                                model, self.questionnaire, persona
                            ),
                            total=len(self.questionnaire),
                            desc=f"{(model_id, persona.id)}",
                        )
                    ],
                    open(prod_path / f"{uuid.uuid4()}.json", "w", encoding="utf8"),
                    indent=4,
                    ensure_ascii=False,
                )
            else:
                logging.info(
                    f"Generated {self.iterations} surveys for configuration: {model_id}:{persona.id}"
                )

        logging.info("> Collate Data")
        survey: data.Survey = data.Survey.from_samples(f"{self.export_path}/**/*.json")
        survey.data.to_parquet(f"{self.export_path}/survey.parquet")

        logging.info("> Write Data Report")
        survey.write_report(f"{self.export_path}/survey.report.txt")

        logging.info("> Generate Cross Evaluation")
        cross_evaluation: data.CrossEvaluation = data.CrossEvaluation.from_survey(
            survey, self.questionnaire.get_survey("graham_et_al")
        )
        cross_evaluation.data.to_parquet(f"{self.export_path}/cross_evaluation.parquet")

    @staticmethod
    def answer_questionnaire(
        model: str,
        questionnaire: Questionnaire,
        persona: Persona,
    ) -> typing.Iterator[typing.Dict]:
        for segment in questionnaire.segments:
            for question in segment.questions:
                response: inference.schemas.Chat = inference.Pipeline(model=model)(
                    inference.schemas.Chat(
                        messages=[
                            inference.schemas.Message(
                                role="system", content=str(persona.content)
                            ),
                            inference.schemas.Message(
                                role="user",
                                content=f"{segment.task}\n\nSentence: {question.content}",
                            ),
                        ]
                    )
                )

                extracted_response: typing.Match | None = re.search(
                    r"(\d)", response[-1].content
                )

                yield {
                    "segment": segment.label,
                    "id": question.id,
                    "dimension": question.dimension,
                    "model": model,
                    "persona": persona.id,
                    "response": extracted_response.group(1)
                    if extracted_response
                    else None,
                }
