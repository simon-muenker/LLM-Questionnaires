import typing

import rich

import llm_questionnaires as llmq


def test_agent(
    personas: typing.List[llmq.agent.AgentPersona],
    models: typing.List[llmq.agent.AgentModel],
):
    rich.print(personas)
    rich.print(models)

    agent = llmq.agent.Agent(
        persona=personas[0],
        model=models[0],
    )

    rich.print(agent("Hello World"))
    rich.print(agent("What is 1+1?", result_type=int))
