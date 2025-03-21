import pathlib

import rich

import llm_questionnaires as llmq

EXP_PATH: str = "experiments/moral_foundations_2/02--naive-personas"


models = [
    llmq.agent.AgentModel(id="llama3.1:8b"),
    llmq.agent.AgentModel(id="mistral:7b"),
    llmq.agent.AgentModel(id="qwen2.5:72b"),
    llmq.agent.AgentModel(id="llama3.3:70b"),
    llmq.agent.AgentModel(id="mistral-large:123b"),
    llmq.agent.AgentModel(id="qwen2.5:7b"),
]
rich.print(models)

personas = llmq.agent.AgentPersona.from_directory(pathlib.Path(f"{EXP_PATH}/personas/"))
rich.print(personas)

questionnaire = llmq.Questionnaire(
    path=pathlib.Path("data/moral_foundations_2/questionnaire.json")
)
rich.print(questionnaire)

pipeline = llmq.Pipeline(
    iterations=50,
    personas=personas,
    models=models,
    questionnaire=questionnaire,
    experiment_path=pathlib.Path(EXP_PATH),
)
pipeline()
