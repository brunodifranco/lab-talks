import plotly.express as px
from pandas.core.frame import DataFrame


def time_series_plot(
    df: DataFrame, target_col: str, title: str, width: int = 1200, height: int = 500
):
    fig = px.line(df, x="Datetime", y=target_col, title=title)
    fig.update_layout(
        xaxis_title="Date", yaxis_title=target_col, width=width, height=height
    )
    fig.show()


def time_series_outlier_plot(
    df: DataFrame, target_col: str, width: int = 1200, height: int = 500
):

    fig = px.line(
        df, x="Datetime", y=target_col, title=f"{target_col} with Outliers Highlighted"
    )

    # Adicionar os outliers em vermelho
    fig.add_scatter(
        x=df.loc[df["Outlier"], "Datetime"],
        y=df.loc[df["Outlier"], target_col],
        mode="markers",
        marker=dict(color="red"),
        name="Outliers",
    )

    fig.update_yaxes(title_text=target_col)

    fig.update_layout(
        xaxis_title="Date", yaxis_title=target_col, width=width, height=height
    )

    fig.show()
