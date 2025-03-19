import typing
import glob
import json

import pandas

INDEX: typing.List[str] = ["segment", "id", "model", "persona"]

class PostProcess:

    def __call__(
        data_pattern: str, 
        model_order: typing.List[str],
        persona_order: typing.List[str]
    ) -> pandas.DataFrame:
        data: pandas.DataFrame = PostProcess.collate_from_raw_data(
            data_pattern, model_order, persona_order
        )

    @staticmethod
    def collate_from_raw_data(
            data_pattern: str, 
            model_order: typing.List[str],
            persona_order: typing.List[str]
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
            df
            .groupby(grouper, observed=True)
            ["response"]
            .agg(["mean", "std"])
            .assign(export=lambda df: formatter(df["mean"]) + " (± " + formatter(df["std"]) + ")")
        )
    