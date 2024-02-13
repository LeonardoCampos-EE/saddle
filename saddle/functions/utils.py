import pandas as pd


def clip_dataframe(
    df: pd.DataFrame, upper: pd.Series, lower: pd.Series,
) -> pd.DataFrame:
    return df.clip(lower=lower, upper=upper, axis=1)  # type: ignore
