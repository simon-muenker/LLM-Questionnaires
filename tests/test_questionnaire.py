import rich

import llm_questionnaires as llmq

questionnaire = llmq.Questionnaire(path="data/moral_foundations_2/questionnaire.json")

for segment in questionnaire.segments:
    rich.print(segment)