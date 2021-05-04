import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Embedding, SpatialDropout1D, Bidirectional, Activation
from tensorflow.keras.models import Sequential


def simple_rnn(vocabulary_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation='sigmoid')
        ])
    return model

def unidirectional_LSTM(vocabulary_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM (64, return_sequences=False),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation='sigmoid')
        ])
    return model

def unidirectional_LSTM(vocabulary_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GRU (64, return_sequences=False),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation='sigmoid')
        ])
    return model

def bi_directional_rnn_lstm(vocabulary_size, embedding_dim, max_length):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embedding_dim, input_length=max_length),
        tf.keras.layers.SpatialDropout1D(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (32)),
        tf.keras.layers.Dense (32, activation='relu'),
        tf.keras.layers.Dense (5, activation='sigmoid')
        ])
    return model


def simple_rnn_glove(vocabulary_size, embedding_dim, max_length, embedding_matrix):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocabulary_size, embedding_dim, input_length = max_length, weights=[embedding_matrix], trainable=False),
    #weights=[embedding_matrix], trainable=False
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(5, activation ='sigmoid')
    ])
    return model

def unidirectional_LSTM(vocabulary_size, embedding_dim, max_length, embedding_matrix):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
        tf.keras.layers.LSTM (64, return_sequences=False),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation='sigmoid')
        ])
    return model

def unidirectional_LSTM(vocabulary_size, embedding_dim, max_length, embedding_matrix):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
        tf.keras.layers.GRU (64, return_sequences=False),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation='sigmoid')
        ])
    return model

def bi_directional_rnn_lstm_glove(vocabulary_size, embedding_dim, max_length, embedding_matrix):

    model = tf.keras.Sequential ([
    tf.keras.layers.Embedding (vocabulary_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM (32)),
    tf.keras.layers.Dense (32, activation='relu'),
    tf.keras.layers.Dense (5, activation='sigmoid')
    ])
    return model


if __name__ == '__main__':
    print ("Modules have been downloaded from %s!" %__name__ )
