import rich

import llm_questionnaires as llmq


def test_questionnaire(questionnaire: llmq.Questionnaire):
    for segment in questionnaire.segments:
        rich.print(segment)
