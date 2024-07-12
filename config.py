import pathlib
import typing

import pydantic

import src

MODELS: typing.Set[str] = {
    'gemma:7b-instruct-q6_K',
    'llama2:70b-chat-q6_K',
    'llama3:70b-instruct-q6_K',
    'mistral:7b-instruct-v0.2-q6_K',
    'mixtral:8x22b-instruct-v0.1-q6_K',
    'mixtral:8x7b-instruct-v0.1-q6_K',
    'qwen:72b-chat-v1.5-q6_K'
}


class Config(pydantic.BaseModel):
    data_dir: pathlib.Path = pathlib.Path("./data/")
    report_dir: pathlib.Path = pathlib.Path("./report/")

    models: typing.Dict[str, str] = {}
    personas: typing.Dict[str, src.Persona] = {}
    questionnaires: typing.Dict[str, src.Questionnaire] = {}

    def model_post_init(self, __context: typing.Any) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.models = {
            (
                model_url
                .split('-')[0]
                .replace(":", "-")
            ): model_url
            for model_url in MODELS
        }

        self.personas = self.load_datafiles(
            self.data_dir / "personas",
            src.Persona.from_json
        )

        self.questionnaires = self.load_datafiles(
            self.data_dir / "questionnaires",
            src.Questionnaire
        )

    @staticmethod
    def load_datafiles(path: pathlib.Path, data_init: typing.Callable) -> typing.Dict[str, typing.Any]:
        return {
            (
                data_path.name
                .replace(".json", "")
                .replace(".", "-")
            ): data_init(path=data_path)
            for data_path in list(path.iterdir())
            if str(data_path.name)[0] != "."
        }

    def __repr__(self) -> str:
        return (
            f"{self.models=}"
            "\n\n"
            f"{set(self.personas.keys())=}"
            "\n\n"
            f"{set(self.questionnaires.keys())=}"
        )
