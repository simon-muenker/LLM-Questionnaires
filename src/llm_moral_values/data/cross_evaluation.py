import typing

import matplotlib.pyplot as plt
import pandas as pd
import pydantic
import seaborn as sns

from llm_moral_values.data.survey import Survey


class CrossEvaluation(pydantic.BaseModel):
    data: pd.DataFrame

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_survey(cls, dataset: Survey, questionnaire_survey) -> typing.Self:
        human_cross_evaluation: typing.List[typing.Dict] = []

        for group_label, group in questionnaire_survey.groups.items():
            for human_label, human in group.items():
                for model_label, model in (
                    dataset.aggregate_by_group(["model", "persona", "dimension"]).T.loc[("mean",)].iterrows()
                ):
                    row = pd.Series(
                        {
                            "sample": group_label,
                            "group": human_label,
                            "model": model_label[0],
                            "persona": model_label[1],
                            "value": None,
                        }
                    )

                    filter_model_data = model[model.index != "catch"]

                    if None not in filter_model_data.to_dict().values():
                        row["value"] = sum(
                            [abs(value - filter_model_data.to_dict()[keys]) for keys, value in human.items()]
                        ) / len(filter_model_data)

                    human_cross_evaluation.append(row)

        return cls(
            data=(
                pd.DataFrame(human_cross_evaluation)
                .pivot(
                    index=("model", "persona"),
                    columns=("sample", "group"),
                    values="value",
                )
                .sort_index()
                .reindex(["base", "liberal", "moderate", "conservative"], axis=0, level=1)
            )
        )

    def plot(
        self,
        export_path: str,
    ):
        fig, ax = plt.subplots(figsize=(10, int(len(self.data) * 0.325)))

        sns.heatmap(self.data, annot=True, fmt=".3f", cmap="crest")

        level_0_values: typing.List[str] = list(self.data.index.get_level_values(0))
        level_1_values: typing.List[str] = list(self.data.index.get_level_values(1))

        ax.hlines(range(0, len(self.data), len(set(level_1_values))), *ax.get_xlim(), linewidth=3.0, color="white")
        ax.vlines(
            range(0, len(self.data.columns), 3),
            *ax.get_ylim(),
            linewidth=3.0,
            color="white",
        )

        secx = ax.secondary_xaxis(location="top")
        secx.set_xticks(
            [1.5, 4.5, 7.5],
            labels=["anonymous\n\n\n\n", "U.S.\n\n\n\n", "Korea\n\n\n\n"],
        )
        secx.tick_params(axis="x", labelsize="large")

        ax.set(xlabel="", ylabel="")

        ax.xaxis.tick_top()
        ax.set_xticklabels(["liberal", "moderate", "conservative"] * 3)
        ax.tick_params(axis="x", labelrotation=45)

        ax.set_yticklabels(level_1_values)

        secy = ax.secondary_yaxis(location="left")
        secy.set_yticks(
            range(2, len(self.data) + 1, len(set(level_1_values))),
            labels=[f"{label}{" " * 24}" for label in set(level_0_values)],
        )
        secy.tick_params(axis="y", color="white", labelsize="large")

        fig.savefig(export_path, bbox_inches="tight")
