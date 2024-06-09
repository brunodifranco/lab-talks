import numpy as np
from pandas.core.frame import DataFrame
from sklearn.ensemble import IsolationForest
from keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
    Dropout,
    Bidirectional,
)
from keras import Model
from keras.callbacks import EarlyStopping
from plots import time_series_outlier_plot


# Primeiro método - Estatístico
def statistical_method(df: DataFrame, target_col: str, threshold: int):
    mean = df[target_col].mean()
    std_dev = df[target_col].std()

    upper_limit = mean + threshold * std_dev
    lower_limit = mean - threshold * std_dev

    # Identificar os outliers
    df["Outlier"] = (df[target_col] > upper_limit) | (df[target_col] < lower_limit)

    time_series_outlier_plot(df=df, target_col=target_col)


# Segundo método - Isolation Forest
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


# Terceiro método - LSTM Autoencoder
class LSTMAutoencoder:
    def __init__(
        self,
        df: DataFrame,
        target_col: str,
        sequence_length: int,
        lstm_layers: int,
        latent_dim: int,
        dropout: float,
        epochs: int,
        batch_size: int,
        threshold_quantile: float,
        early_stopping_patience: int = 10,
    ):
        self.df = df
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.lstm_layers = lstm_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_quantile = threshold_quantile
        self.early_stopping_patience = early_stopping_patience

    def preprocessing(self) -> np.ndarray:
        X = np.array(
            [
                self.df[self.target_col].values[i : i + self.sequence_length]
                for i in range(len(self.df) - self.sequence_length)
            ]
        )
        X = np.expand_dims(X, axis=-1)

        return X

    def build_model(self, X: np.ndarray) -> Model:

        # X
        input_dim = X.shape[2]
        timesteps = X.shape[1]

        # Encoder
        inputs = Input(shape=(timesteps, input_dim))
        encoder = Bidirectional(
            LSTM(self.lstm_layers, activation="relu", return_sequences=False)
        )(inputs)
        encoder = Dropout(self.dropout)(encoder)

        # Latent
        latent = Dense(self.latent_dim, activation="relu")(encoder)
        repeat = RepeatVector(timesteps)(latent)

        # Decoder
        decoder = Bidirectional(
            LSTM(self.lstm_layers, activation="relu", return_sequences=True)
        )(repeat)
        decoder = Dropout(self.dropout)(decoder)
        output = TimeDistributed(Dense(input_dim))(decoder)

        model = Model(inputs, output)

        return model

    def train(self, model: Model, X: np.ndarray):
        model.compile(optimizer="adam", loss="mse")

        # Set callback
        early_stopping = EarlyStopping(
            monitor="loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True,
        )
        history = model.fit(
            X,
            X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
        )

        return history

    def get_anomalies(self, X: np.ndarray, model: Model) -> DataFrame:

        reconstructions = model.predict(X)
        reconstruction_errors = np.mean(np.square(reconstructions - X), axis=(1, 2))

        self.df["reconstruction_error"] = np.concatenate(
            [np.zeros(self.sequence_length), reconstruction_errors]
        )
        threshold = self.df["reconstruction_error"].quantile(self.threshold_quantile)
        self.df["Outlier"] = self.df["reconstruction_error"] > threshold

        return self.df

    def plot_anomalies(self):

        time_series_outlier_plot(self.df, self.target_col)
