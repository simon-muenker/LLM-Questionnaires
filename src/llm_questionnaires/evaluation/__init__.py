import typing
import json
import glob

import numpy

import pandas
import pandas.io.formats
import pandas.io.formats.style 


def read_populations(path: str, populations: typing.List[str], columns: typing.List[str]) -> typing.Dict[str, pandas.DataFrame]:
    return {
        population: (
                pandas.concat(
                [
                    pandas.json_normalize(json.load(open(file))).assign(participant=n)
                    for n, file in enumerate(glob.glob(f"{path}/{population}/*.json", recursive=True))
                ],
            )
            .pivot(index="participant", columns="id", values="response")
            .apply(pandas.to_numeric)
            .set_axis(columns, axis=1)
        )
        for population in populations
    }


def calc_fingerprint(df: pandas.DataFrame) -> numpy.ndarray:
    corr_matrix = numpy.triu(df.corr("pearson").fillna(0.0).to_numpy())

    return corr_matrix[numpy.triu_indices_from(corr_matrix, k=1)]


def calc_similarity(df_1: pandas.DataFrame, df_2: pandas.DataFrame) -> float:
    x1 = calc_fingerprint(df_1)
    x2 = calc_fingerprint(df_2)
    
    return numpy.dot(x1, x2)/(numpy.linalg.norm(x1)*numpy.linalg.norm(x2))


def apply_calc_similarity(populations: typing.Dict[str, pandas.DataFrame]) -> pandas.DataFrame:
    return (
        pandas.concat({
            (key_1, key_2): pandas.Series(
                calc_similarity(values_1, values_2),
                name="similarity"
            )
            for key_1, values_1 in populations.items()
            for key_2, values_2 in populations.items()
        })
    )


def format_latex_df(df: pandas.DataFrame) -> pandas.io.formats.style.Styler:
    return (
        df
        .style
        .format(na_rep="", precision=3) 
        .format_index(escape="latex", axis=1)
        .format_index(escape="latex", axis=0)
        .map_index(
            lambda v: "rotatebox:{65}--rwrap--latex;", level=0, axis=1
        ) 
        .to_latex(convert_css=True, hrules=True, clines="skip-last;data") 
    )


def extract_triu_df(df: pandas.DataFrame) -> pandas.DataFrame:
    return (
        df
        .where(
            pandas.DataFrame(
                [[i > j for j in range(len(df.columns))] for i in range(len(df.index))],
                index=df.index,
                columns=df.columns
            )
        )
    )