from sklearn.ensemble import IsolationForest
from pandas.core.frame import DataFrame
from plots import time_series_outlier_plot


def statistical_method(df: DataFrame, target_col: str, threshold: int):
    mean = df[target_col].mean()
    std_dev = df[target_col].std()

    upper_limit = mean + threshold * std_dev
    lower_limit = mean - threshold * std_dev

    # Identificar os outliers
    df["Outlier"] = (df[target_col] > upper_limit) | (df[target_col] < lower_limit)

    time_series_outlier_plot(df=df, target_col=target_col)


def isolation_forest_method(
    df: DataFrame, target_col: str, n_estimators: int, contamination: float
):
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    df["anomaly"] = iso_forest.fit_predict(df[[target_col]])

    # Marcar os outliers
    df["Outlier"] = df["anomaly"] == -1

    time_series_outlier_plot(df=df, target_col=target_col)
