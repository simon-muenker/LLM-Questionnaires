import typing

import pandas
import pydantic
import seaborn


class DescriptiveAnalysis(pydantic.BaseModel):
    data: pandas.DataFrame

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.computed_field
    @property
    def data_wo_idx(self) -> pandas.DataFrame:
        return self.data.reset_index()

    @pydantic.computed_field
    @property
    def plot_styles(self) -> typing.Dict:
        return dict(
            linestyle="none",
            palette=seaborn.color_palette("husl", len(self.data_wo_idx["persona"].unique())),
            dodge=(0.8 - 0.8 / len(self.data_wo_idx["persona"].unique())),
            capsize=0.1,
            markersize=1.4,
            err_kws={"linewidth": 1, "alpha": 0.3},
        )

    def write_report(self, export_file: str) -> None:
        with open(export_file, "w") as f:
            for model, group in (
                self.data.groupby(self.data.index.names, observed=True)["response"]
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

    def plot(self, export_path: str):
        grid = seaborn.FacetGrid(
            (
                self.data_wo_idx[self.data_wo_idx["dimension"] != "catch"]
                .pipe(
                    lambda _df: _df.assign(
                        dimension=_df["dimension"].cat.remove_unused_categories()
                    )
                )
                .reset_index()
            ),
            col="model",
            col_wrap=3,
            height=max(3, len(self.data_wo_idx["persona"].unique())),
        )

        grid.map_dataframe(
            seaborn.pointplot,
            x="response",
            y="dimension",
            hue="persona",
            errorbar="sd",
            **self.plot_styles,
        )
        grid.add_legend()
        grid.savefig(export_path)
