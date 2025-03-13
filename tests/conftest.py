import pathlib
import typing

import pytest

import llm_questionnaires as llmq


@pytest.fixture(scope="session")
def model_selection() -> typing.List[str]:
    return ["llama3.1:8b", "llama3.3:70b"]


@pytest.fixture(scope="session")
def persona_dir() -> pathlib.Path:
    return pathlib.Path("tests/fixtures/personas/")


@pytest.fixture(scope="session")
def questionnaire_path() -> pathlib.Path:
    return pathlib.Path("tests/fixtures/questionnaire.json")


@pytest.fixture
def personas(persona_dir: pathlib.Path) -> typing.List[llmq.agent.AgentPersona]:
    return llmq.agent.AgentPersona.from_directory(persona_dir)


@pytest.fixture
def models(model_selection: typing.List[str]) -> typing.List[llmq.agent.AgentModel]:
    return [llmq.agent.AgentModel(id=model) for model in model_selection]


@pytest.fixture
def questionnaire(questionnaire_path: pathlib.Path) -> llmq.Questionnaire:
    return llmq.Questionnaire(path=questionnaire_path)
