import pathlib

import rich

import llm_questionnaires as llmq

EXP_PATH: str = "experiments/moral_foundations/03--value-driven-personas"
PERSONA_ORDER = ["liberal", "moderate", "conservative"]

models = [
    llmq.agent.AgentModel(id="llama3.1:8b"),
    llmq.agent.AgentModel(id="llama2:70b"),
    llmq.agent.AgentModel(id="llama3:70b"),
    llmq.agent.AgentModel(id="llama3.1:70b"),
    llmq.agent.AgentModel(id="mistral:7b"),
    llmq.agent.AgentModel(id="mixtral:8x22b"),
    llmq.agent.AgentModel(id="mixtral:8x7b"),
    llmq.agent.AgentModel(id="phi3:14b"),
    llmq.agent.AgentModel(id="gemma:7b"),
    llmq.agent.AgentModel(id="gemma2:27b"),
    llmq.agent.AgentModel(id="qwen:72b"),
    llmq.agent.AgentModel(id="qwen2:72b"),
]
rich.print(models)

personas = llmq.agent.AgentPersona.from_directory(pathlib.Path(f"{EXP_PATH}/personas/"))
rich.print(personas)

questionnaire = llmq.Questionnaire(
    path=pathlib.Path("data/moral_foundations/questionnaire.json")
)
rich.print(questionnaire)

pipeline = llmq.Pipeline(
    iterations=50,
    personas=personas,
    models=models,
    questionnaire=questionnaire,
    export_path=pathlib.Path(f"{EXP_PATH}/data/"),
)

pipeline()