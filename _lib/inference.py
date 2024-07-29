import logging
import re
import typing

import pydantic
import requests

import ollama


class InferenceResponse(pydantic.BaseModel):
    value: str | None

    def extract_numeric_answer(self) -> int | None:
        if not self.value:
            return None

        groups: typing.Match | None = re.search(r'(\d)', self.value)
        return groups.group(1) if groups else None


class Inference(pydantic.BaseModel):
    model: str
    system_prompt: str | None = None

    remote_endpoint: str = "https://inf.cl.uni-trier.de/"

    def __call__(self, prompt: str, use_remote: bool = True):
        return InferenceResponse(value=(
            self._remote(prompt)
            if use_remote else
            self._local(prompt)
        ))
    
    def _remote(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.remote_endpoint,
                json={
                    "model": self.model,
                    "system": self.system_prompt if self.system_prompt else None,
                    "prompt": prompt
                }
            ).json()["response"]

        except Exception as e:
            logging.warning(e)

        finally:
            return response

    def _local(self, prompt: str) -> str:
        return ollama.generate(
            model=self.model,
            system=self.system_prompt,
            prompt=prompt
        )["response"]