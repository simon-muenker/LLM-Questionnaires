import glob
import json
import pathlib
import typing

import pandas

INDEX: typing.List[str] = ["segment", "id", "model", "persona"]


class PostProcess:
    def __call__(
        self,
        data_pattern: str,
        model_order: typing.List[str],
        persona_order: typing.List[str],
        export_path: pathlib.Path,
    ) -> pandas.DataFrame:
        dataset: pandas.DataFrame = PostProcess.collate_from_raw_data(
            data_pattern, model_order, persona_order
        )

        dataset.to_parquet(f"{export_path}/dataset.long.parquet")
        dataset.to_csv(f"{export_path}/dataset.long.csv")

        dataset_agg_idx_pvt_mod: pandas.DataFrame = (
            PostProcess()
            .aggregate(dataset, ["id", "model", "persona"])
            .pivot_table(
                index=["id", "persona"],
                columns=["model"],
                values="export",
                aggfunc="first",
                observed=True,
            )
            .sort_index(level=0, axis=1)
        )

        dataset_agg_idx_pvt_mod.to_csv(f"{export_path}/dataset._agg_idx.pvt_mod.csv")
        dataset_agg_idx_pvt_mod.to_latex(
            f"{export_path}/dataset._agg_idx.pvt_mod.tex", bold_rows=True
        )

        dataset_agg_dim_pvt_dim: pandas.DataFrame = (
            PostProcess()
            .aggregate(dataset, ["model", "persona", "dimension"])
            .pivot_table(
                index=["model", "persona"],
                columns=["dimension"],
                values="export",
                aggfunc="first",
                observed=True,
            )
            .swaplevel(0, 1, axis=0)
            .sort_index(level=0)
        )

        dataset_agg_dim_pvt_dim.to_csv(f"{export_path}/dataset.agg_dim.pvt_dim.csv")
        dataset_agg_dim_pvt_dim.to_latex(
            f"{export_path}/dataset.agg_dim.pvt_dim.tex", bold_rows=True
        )

        return dataset

    @staticmethod
    def collate_from_raw_data(
        data_pattern: str, model_order: typing.List[str], persona_order: typing.List[str]
    ) -> pandas.DataFrame:
        return (
            pandas.concat(
                [
                    pandas.json_normalize(json.load(open(file)))
                    for file in glob.glob(data_pattern, recursive=True)
                ]
            )
            .pipe(
                lambda _df: _df.assign(
                    segment=_df["segment"].astype("category"),
                    model=pandas.Categorical(_df["model"], categories=model_order),
                    persona=pandas.Categorical(_df["persona"], categories=persona_order),
                    response=pandas.to_numeric(_df["response"]),
                    id=pandas.to_numeric(_df["id"]),
                    dimension=_df["dimension"].astype("category"),
                )
            )
            .set_index(INDEX)
            .sort_index()
            .dropna()
        )

    @staticmethod
    def aggregate(
        df: pandas.DataFrame,
        grouper: typing.List[str],
    ) -> pandas.DataFrame:
        def formatter(row: pandas.Series, precision: int = 2) -> pandas.Series:
            return row.round(precision).astype(str).str.ljust(precision + 2, fillchar="0")

        return (
            df.groupby(grouper, observed=True)["response"]
            .agg(["mean", "std"])
            .assign(
                export=lambda df: formatter(df["mean"]) + " (± " + formatter(df["std"]) + ")"
            )
        )
