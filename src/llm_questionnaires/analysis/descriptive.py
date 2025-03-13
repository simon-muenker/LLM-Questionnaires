import glob
import json
import typing

import pandas
import pydantic
import seaborn


class DescriptiveAnalysisArgs(pydantic.BaseModel):
    model_order: typing.List[str]
    persona_order: typing.List[str]

    index: typing.List[str] = ["segment", "id", "model", "persona"]

    model_config = pydantic.ConfigDict(protected_namespaces=())

    @pydantic.computed_field
    @property
    def plot_styles(self) -> typing.Dict:
        return dict(
            linestyle="none",
            palette=seaborn.color_palette("husl", len(self.persona_order)),
            dodge=(0.8 - 0.8 / len(self.persona_order)),
            capsize=0.1,
            markersize=1.4,
            err_kws={"linewidth": 1, "alpha": 0.3},
        )


class DescriptiveAnalysis(pydantic.BaseModel):
    data: pandas.DataFrame
    args: DescriptiveAnalysisArgs

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_samples(
        cls, raw_samples_pattern: str, args: DescriptiveAnalysisArgs
    ) -> "DescriptiveAnalysis":
        return cls(
            data=(
                pandas.concat(
                    [
                        pandas.json_normalize(json.load(open(file)))
                        for file in glob.glob(raw_samples_pattern, recursive=True)
                    ]
                )
                .pipe(
                    lambda _df: _df.assign(
                        segment=_df["segment"].astype("category"),
                        model=pandas.Categorical(_df["model"], categories=args.model_order),
                        persona=pandas.Categorical(
                            _df["persona"], categories=args.persona_order
                        ),
                        response=pandas.to_numeric(_df["response"]),
                        id=pandas.to_numeric(_df["id"]),
                        dimension=_df["dimension"].astype("category"),
                    )
                )
                .set_index(args.index)
                .sort_index()
                .dropna()
            ),
            args=args,
        )

    def write_report(self, export_file: str) -> None:
        with open(export_file, "w") as f:
            for model, group in (
                self.data.groupby(self.args.index, observed=True)["response"]
                .var()
                .groupby("model", observed=False)
            ):
                f.write(f"{model:-^42}\n")
                f.write(f"answers w/o variance: {len(group[group == 0.0])}/{len(group)}\n")
                f.write(f"mean variance: {group.mean():2.3f}\n")
                f.write("answers with variance (10 decile):\n")
                f.write(
                    f"{group[group > group.quantile(0.9)].droplevel(2).sort_values(ascending=False).to_string()}\n"
                )
                f.write("\n\n")

    def aggregate_by_group(
        self,
        group: typing.List,
        index: typing.List = ["model", "persona"],
    ) -> pandas.DataFrame:
        return (
            self.data.groupby(group, observed=True)["response"]
            .agg(mean="mean", var="var")
            .sort_index()
            .reset_index(index)
            .pivot(columns=index)
        )

    def plot(self, export_path: str):
        grid = seaborn.FacetGrid(
            (
                self.data[self.data["dimension"] != "catch"]
                .pipe(
                    lambda _df: _df.assign(
                        dimension=_df["dimension"].cat.remove_unused_categories()
                    )
                )
                .reset_index()
            ),
            col="model",
            col_wrap=4,
            height=len(self.args.persona_order),
        )

        grid.map_dataframe(
            seaborn.pointplot,
            x="response",
            y="dimension",
            hue="persona",
            errorbar="sd",
            **self.args.plot_styles,
        )
        grid.add_legend()
        grid.savefig(export_path)
