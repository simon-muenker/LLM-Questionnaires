import glob
import itertools
import json
import pathlib
import typing
import uuid

import pandas
import pydantic
from rich.progress import track

from llm_questionnaires.agent import Agent, AgentModel, AgentPersona
from llm_questionnaires.data.descriptive_analysis import DescriptiveAnalysis
from llm_questionnaires.data.postprocess import PostProcess
from llm_questionnaires.questionnaire import Questionnaire


class Pipeline(pydantic.BaseModel):
    iterations: int

    personas: typing.List[AgentPersona]
    models: typing.List[AgentModel]
    questionnaire: Questionnaire

    experiment_path: pathlib.Path

    _data_path: pathlib.Path
    _report_path: pathlib.Path

    def model_post_init(self, _: typing.Any):
        self._data_path = self.experiment_path / "data"
        self._report_path = self.experiment_path / "reports"

        self._data_path.mkdir(parents=True, exist_ok=True)
        self._report_path.mkdir(parents=True, exist_ok=True)

    def __call__(self):
        for persona, model in list(itertools.product(self.personas, self.models)):
            self.process_configuration(persona=persona, model=model)

        dataset: pandas.DataFrame = PostProcess()(
            data_pattern=f"{self._data_path}/**/*.json",
            model_order=[model.id for model in self.models],
            persona_order=[persona.id for persona in self.personas],
            export_path=self._report_path,
        )

        descriptive_analysis = DescriptiveAnalysis(data=dataset)
        descriptive_analysis.write_report(f"{self._report_path}/report.variance.by_model.txt")
        descriptive_analysis.plot(f"{self._report_path}/point.dimensions.by_model.pdf")

    def process_configuration(self, persona: AgentPersona, model: AgentModel) -> None:
        iteration_path: pathlib.Path = self._data_path / persona.dir_name / model.dir_name
        iteration_path.mkdir(parents=True, exist_ok=True)

        agent = Agent(persona=persona, model=model)
        remaining_survey_num: int = self.iterations - len(glob.glob(f"{iteration_path}/*.json"))

        if remaining_survey_num <= 0:
            return

        for _ in track(
            range(remaining_survey_num),
            description=f"Generating {remaining_survey_num} surveys for ({model.id}|{persona.id}): ",
        ):
            self.conduct_survey(agent, iteration_path)

    def conduct_survey(self, agent: Agent, export_path: pathlib.Path) -> None:
        json.dump(
            [
                {
                    "segment": segment.label,
                    "id": question.id,
                    "dimension": question.dimension,
                    "model": agent.model.id,
                    "persona": agent.persona.id,
                    "response": agent(
                        user_prompt=f"{segment.task}\nQuestion: {question.content}",
                        result_type=typing.Literal[tuple(item for item in segment.scale)],
                    ),
                }
                for segment in self.questionnaire.segments
                for question in segment.questions
            ],
            open(export_path / f"{uuid.uuid4()}.json", "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )
