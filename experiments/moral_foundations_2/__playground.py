import pandas
import rich

PATH_PATH: str = "experiments/moral_foundations_2"

SELECTED_COUNTRIES = [
    "Belgium",
    "France",
    "Egypt",
    "Saudi Arabia",
    "Russia",
    "Mexico",
    "Argentina",
    "Japan",
    "New Zealand",
]


def format_export(df: pandas.DataFrame, precision: int = 2) -> pandas.Series:
    def formatter(row: pandas.Series) -> pandas.Series:
        return row.round(precision).astype(str).str.ljust(precision + 2, fillchar="0")

    return formatter(df["mean"]) + " (± " + formatter(df["std"]) + ")"


raw: pandas.DataFrame = pandas.read_parquet(f"{PATH_PATH}/analysis/survey.parquet")

agg_idx: pandas.DataFrame = (
    raw
    .groupby(["id", "model", "persona"], observed=True)
    ["response"]
    .agg(["mean", "std"])
    .assign(export=lambda df: format_export(df))
)

agg_idx_pvt_mod: pandas.DataFrame = agg_idx.pivot_table(
    index=["id", "persona"], columns=["model"], values="export", aggfunc="first", observed=True
).sort_index(level=0, axis=1)
rich.print(agg_idx_pvt_mod)

agg_dim: pandas.DataFrame = (
    raw.groupby(["model", "persona", "dimension"], observed=True)["response"]
    .agg(["mean", "std"])
    .assign(export=lambda df: format_export(df))
)

agg_dim_pvt_dim: pandas.DataFrame = (
    agg_dim.pivot_table(
        index=["model", "persona"],
        columns=["dimension"],
        values="export",
        aggfunc="first",
        observed=True,
    )
    .swaplevel(0, 1, axis=0)
    .sort_index(level=0)
)
rich.print(agg_dim_pvt_dim)

agg_dim_pvt_dim.to_csv(f"{PATH_PATH}/analysis/agg_dim_pvt_dim.csv")

(
    agg_dim_pvt_dim.to_latex(
        f"{PATH_PATH}/analysis/agg_dim_pvt_dim.tex",
        bold_rows=True,
    )
)
