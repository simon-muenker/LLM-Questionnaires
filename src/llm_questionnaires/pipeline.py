import glob
import itertools
import json
import logging
import pathlib
import typing
import uuid

import pydantic

from llm_questionnaires.agent import Agent, AgentModel, AgentPersona
from llm_questionnaires.questionnaire import Questionnaire


class Pipeline(pydantic.BaseModel):
    iterations: int

    personas: typing.List[AgentPersona]
    models: typing.List[AgentModel]
    questionnaire: Questionnaire

    export_path: pathlib.Path

    def __call__(self):
        logging.info("> Conducting Survey")
        for persona, model in list(itertools.product(self.personas, self.models)):
            self.process_configuration(persona=persona, model=model)

    def process_configuration(self, persona: AgentPersona, model: AgentModel) -> None:
        iteration_path: pathlib.Path = self.export_path / persona.id / model.dir_name
        iteration_path.mkdir(parents=True, exist_ok=True)

        agent = Agent(persona=persona, model=model)

        while len(glob.glob(f"{iteration_path}/*.json")) < self.iterations:
            self.conduct_survey(agent, iteration_path)

        logging.info(
            f"Generated {self.iterations} surveys for configuration: {model.name}:{persona.id}"
        )

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
