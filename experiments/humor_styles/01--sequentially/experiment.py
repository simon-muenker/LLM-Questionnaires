import pathlib

import rich

import llm_questionnaires as llmq

EXP_PATH: str = "experiments/humor_styles/01--baseline"


models = [
    llmq.agent.AgentModel(id="llama3.1:8b"),
    llmq.agent.AgentModel(id="mistral:7b"),
    llmq.agent.AgentModel(id="qwen2.5:7b"),
    llmq.agent.AgentModel(id="llama3.3:70b"),
    llmq.agent.AgentModel(id="mistral-large:123b"),
    llmq.agent.AgentModel(id="qwen2.5:72b"),
]
rich.print(models)

personas = llmq.agent.AgentPersona.from_directory(pathlib.Path(f"{EXP_PATH}/personas/"))
rich.print(personas)

questionnaire = llmq.Questionnaire(
    path=pathlib.Path("data/humor_styles/questionnaire.json")
)
rich.print(questionnaire)

pipeline = llmq.Pipeline(
    iterations=1000,
    personas=personas,
    models=models,
    questionnaire=questionnaire,
    experiment_path=pathlib.Path(EXP_PATH),
)
pipeline()
