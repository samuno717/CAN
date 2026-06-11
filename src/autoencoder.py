from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_autoencoder(feature_count, bottleneck_size):

    model = Sequential([
        Input(shape=(feature_count,)),
        Dense(6, activation='relu'),
        Dense(bottleneck_size, activation='relu'),
        Dense(6, activation='relu'),
        Dense(feature_count, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mse')

    return model
