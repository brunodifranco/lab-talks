import numpy as np
from pandas.core.frame import DataFrame
import tensorflow as tf
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Bidirectional
from keras import Model
from keras.callbacks import EarlyStopping



class LSTMAutoencoder:
    def __init__(self, df: DataFrame,  target_col: str, sequence_length: int, lstm_layers: int, 
                 
                 latent_dim: int, dropout: float, early_stopping_patience: int, 
                 epochs: int, batch_size: int
                 ):
        self.df = df
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.lstm_layers = lstm_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.batch_size = batch_size

    def preprocessing(self) -> np.ndarray:
        X = np.array([self.df[self.target_col].values[i:i+self.sequence_length] for i in range(len(self.df)-self.sequence_length)])
        X = np.expand_dims(X, axis=-1)

        return X

    def build(self, X: np.ndarray):

        input_dim = X.shape[2]
        timesteps = X.shape[1]

        inputs = Input(shape=(timesteps, input_dim))
        encoder = Bidirectional(LSTM(64, activation='relu', return_sequences=False))(inputs)
        encoder = Dropout(0.2)(encoder)
        latent = Dense(4, activation='relu')(encoder)
        repeat = RepeatVector(timesteps)(latent)
        decoder = Bidirectional(LSTM(64, activation='relu', return_sequences=True))(repeat)
        decoder = Dropout(0.2)(decoder)
        output = TimeDistributed(Dense(input_dim))(decoder)

        model = Model(inputs, output)

        return model
    
    def train(self, model: Model, X: np.ndarray):
        model.compile(optimizer='adam', loss='mse')

        # Definir o callback de Early Stopping
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Treinar o modelo
        history = model.fit(X, X, epochs=500, batch_size=32, callbacks=[early_stopping])
