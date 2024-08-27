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
import pandas as pd

from llm_moral_values import inference
from llm_moral_values.questionnaire import Questionnaire
from llm_moral_values.persona import Persona


class ConductSurvey(pydantic.BaseModel):
    iterations: int
    models: typing.List[str]

    questionnaire: Questionnaire
    personas: typing.List[Persona]

    export_path: pathlib.Path

    def __call__(self):
        for model, persona in list(itertools.product(self.models, self.personas)):
            model_id: str = model.split("-")[0].replace(":", "-")

            prod_path: pathlib.Path = self.export_path / persona.id / model_id
            prod_path.mkdir(parents=True, exist_ok=True)

            while len(glob.glob(f"{prod_path}/*.json")) < self.iterations:
                json.dump(
                    [
                        item
                        for item in tqdm.tqdm(
                            self.answer_questionnaire(
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
                                role="system", content=persona.content
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


class EvaluateSurvey:
    @staticmethod
    def collate_data(raw_samples_pattern: str) -> pd.DataFrame:
        return (
            pd.concat(
                [
                    pd.json_normalize(json.load(open(file, "r")))
                    for file in glob.glob(raw_samples_pattern, recursive=True)
                ]
            )
            .pipe(
                lambda _df: _df.assign(
                    segment=_df["segment"].astype("category"),
                    model=_df["model"].astype("category").str.split("-").str[0],
                    response=pd.to_numeric(_df["response"]),
                    id=pd.to_numeric(_df["id"]),
                    dimension=_df["dimension"].astype("category"),
                )
            )
            .set_index(["segment", "id", "model", "persona"])
            .sort_index()
            .dropna()
        )

    @staticmethod
    def get_response_freq(collated_dataset: pd.DataFrame) -> pd.DataFrame:
        return (
            collated_dataset.reset_index()[["model", "persona"]]
            .value_counts()
            .to_frame()
            .sort_index()
            .T
        )

    @staticmethod
    def write_data_report(collated_dataset: pd.DataFrame, export_file: str) -> None:
        with open(export_file, "w") as f:
            for model, group in (
                collated_dataset.groupby(
                    ["segment", "id", "model", "dimension"], observed=True
                )["response"]
                .var()
                .groupby("model", observed=False)
            ):
                f.write(f"{model:-^42}\n")
                f.write(
                    f"answers w/o variance: {len(group[group == 0.0])}/{len(group)}\n"
                )
                f.write(f"mean variance: {group.mean():2.3f}\n")
                f.write("answers with variance:\n")
                f.write(f"{group[group != 0.0].sort_values(ascending=False)}\n")
                f.write("\n\n")

    @staticmethod
    def aggregate_by_group(
        collated_dataset: pd.DataFrame,
        group: typing.List,
        index: typing.List = ["model", "persona"],
    ) -> pd.DataFrame:
        return (
            collated_dataset.groupby(group, observed=True)["response"]
            .agg(mean="mean", var="var")
            .sort_index()
            .reset_index(index)
            .pivot(columns=index)
        )

    @staticmethod
    def perform_cross_evaluation(
        collated_dataset: pd.DataFrame, questionnaire_survey
    ) -> pd.DataFrame:
        human_cross_evaluation: typing.List[typing.Dict] = []

        for group_label, group in questionnaire_survey.groups.items():
            for human_label, human in group.items():
                for model_label, model in (
                    EvaluateSurvey.aggregate_by_group(
                        collated_dataset, ["model", "persona", "dimension"]
                    )
                    .T.loc[("mean",)]
                    .iterrows()
                ):
                    row = pd.Series(
                        {
                            "sample": group_label,
                            "group": human_label,
                            "model": model_label[0],
                            "persona": model_label[1],
                            "value": None,
                        }
                    )

                    model = model[model.index != "catch"]

                    if None not in model[model.index != "catch"].to_dict().values():
                        row["value"] = sum(
                            [
                                abs(value - model.to_dict()[keys])
                                for keys, value in human.items()
                            ]
                        ) / len(model)

                    human_cross_evaluation.append(row)

        return (
            pd.DataFrame(human_cross_evaluation)
            .pivot(
                index=("model", "persona"), columns=("sample", "group"), values="value"
            )
            .sort_index()
            .reindex(["base", "liberal", "moderate", "conservative"], axis=0, level=1)
        )
