import pathlib
import typing

import llm_questionnaires as llmq


def test_pipeline(
    personas: typing.List[llmq.agent.AgentPersona],
    models: typing.List[llmq.agent.AgentModel],
    questionnaire: llmq.Questionnaire,
):
    pipeline = llmq.Pipeline(
        iterations=2,
        personas=personas,
        models=models,
        questionnaire=questionnaire,
        export_path=pathlib.Path("tests/output/"),
    )
