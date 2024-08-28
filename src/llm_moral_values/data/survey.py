import typing
import glob
import json

import pydantic
import pandas as pd
import seaborn as sns


class Survey(pydantic.BaseModel):
    data: pd.DataFrame

    index: typing.ClassVar[typing.List[str]] = ["segment", "id", "model", "persona"]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_samples(cls: "Survey", raw_samples_pattern: str) -> pd.DataFrame:
        return cls(
            data=(
                pd.concat(
                    [
                        pd.json_normalize(json.load(open(file, "r")))
                        for file in glob.glob(raw_samples_pattern, recursive=True)
                    ]
                )
                .pipe(
                    lambda _df: _df.assign(
                        segment=_df["segment"].astype("category"),
                        model=_df["model"].astype("category").str.split("-").str[0],
                        response=pd.to_numeric(_df["response"]),
                        id=pd.to_numeric(_df["id"]),
                        dimension=_df["dimension"].astype("category"),
                    )
                )
                .set_index(cls.index)
                .sort_index()
                .dropna()
            )
        )

    def write_report(self, export_file: str) -> None:
        with open(export_file, "w") as f:
            for model, group in (
                self.data.groupby(self.index, observed=True)["response"].var().groupby("model", observed=False)
            ):
                f.write(f"{model:-^42}\n")
                f.write(f"answers w/o variance: {len(group[group == 0.0])}/{len(group)}\n")
                f.write(f"mean variance: {group.mean():2.3f}\n")
                f.write("answers with variance (10 decile):\n")
                f.write(f"{group[group > group.quantile(.9)].droplevel(2).sort_values(ascending=False).to_string()}\n")
                f.write("\n\n")

    def aggregate_by_group(
        self,
        group: typing.List,
        index: typing.List = ["model", "persona"],
    ) -> pd.DataFrame:
        return (
            self.data.groupby(group, observed=True)["response"]
            .agg(mean="mean", var="var")
            .sort_index()
            .reset_index(index)
            .pivot(columns=index)
        )

    def plot(
        self,
        export_path: str,
        model_order: typing.List[str],
        persona_order: typing.List[str],
    ):
        grid = sns.FacetGrid(
            (
                self.data[self.data["dimension"] != "catch"]
                .pipe(lambda _df: _df.assign(dimension=_df["dimension"].cat.remove_unused_categories()))
                .reset_index()
            ),
            col="model",
            col_wrap=4,
            col_order=model_order,
            height=4,
        )

        grid.map_dataframe(
            sns.pointplot,
            errorbar="sd",
            x="response",
            y="dimension",
            hue="persona",
            linestyle="none",
            hue_order=persona_order,
            palette=sns.color_palette()[:len(persona_order)],
            dodge=(0.8 - 0.8 / len(persona_order)),
            capsize=0.1,
            markersize=1.4,
            err_kws={"linewidth": 1, "alpha": 0.3},
        )
        grid.add_legend()
        grid.savefig(export_path)
