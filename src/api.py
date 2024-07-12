import logging
import re
import typing

import pydantic
import requests


class InferenceResponse(pydantic.BaseModel):
    value: str | None

    def extract_numeric_answer(self) -> int | None:
        if not self.value:
            return None

        groups: typing.Match | None = re.search(r'(\d)', self.value)
        return groups.group(1) if groups else None


class Inference(pydantic.BaseModel):
    model: str = "mixtral:8x22b-instruct-v0.1-q6_K"
    system_prompt: str | None = None

    endpoint: str = "https://inf.cl.uni-trier.de/"

    def __call__(self, prompt: str):
        request_obj: typing.Dict = {
            "model": self.model,
            "prompt": prompt
        }

        if self.system_prompt:
            request_obj["system"] = self.system_prompt

        response = None

        try:
            response = requests.post(
                self.endpoint,
                json=request_obj
            ).json()["response"]

        except Exception as e:
            logging.warning(e)

        finally:
            return InferenceResponse(value=response)
