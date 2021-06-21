from enum import Enum

from keras import backend as K, Sequential, Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Permute, Dense, multiply, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, TimeDistributed, \
    RepeatVector, Dropout, GRU, AveragePooling1D
from matplotlib import pyplot, pyplot as plt

import enum

class ModelType(enum.Enum):
    SIMPLE_LSTM = 1
    STACKED_LSTM = 2
    BIDRECTIONAL_LSTM = 3
    CNN = 4
    CNN_LSTM = 5
    LSTM_AUTOENCODER = 6
    DEEP_CNN = 7
    GRU = 8
    GRU_CNN = 9



def attention_block(inputs, time_steps):
    x = Permute((2, 1))(inputs)
    x = Dense(time_steps, activation="softmax")(x)
    x = Permute((2, 1), name="attention_prob")(x)
    x = multiply([inputs, x])
    return x


def get_activation(model, layer_name, inputs):
    layer = [l for l in model.layers if l.name == layer_name][0]

    func = K.function([model.input], [layer.output])

    return func([inputs])[0]


def make_model(model_type, Xtrain, Ytrain, opt='adam', loss_func='mse', summary=False, binary=False):
    if binary:
        LAST_ACTIVATION = 'sigmoid'
    else:
        LAST_ACTIVATION = 'linear'
    print(model_type)
    if model_type is ModelType.SIMPLE_LSTM:
        print(model_type)
        # Single cell LSTM
        model = Sequential()
        model.add(LSTM(units=100, activation='relu', name='first_lstm', recurrent_dropout=0.1,
                       input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dense(1, activation=LAST_ACTIVATION))

    elif model_type is ModelType.STACKED_LSTM:
        # Stacked LSTM
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, recurrent_dropout=0.1,
                       input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(LSTM(50, activation='relu', return_sequences=True, recurrent_dropout=0.1))
        model.add(LSTM(30, activation='relu', recurrent_dropout=0.2))
        model.add(Dense(1, activation=LAST_ACTIVATION))

    elif model_type is ModelType.BIDRECTIONAL_LSTM:
        # Bidirectional LSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(100, return_sequences=True, activation='relu')))
        model.add(Bidirectional(LSTM(50, return_sequences=True, activation='relu')))
        model.add(Bidirectional(LSTM(20, activation='relu')))
        model.add(Dense(1, activation=LAST_ACTIVATION))

    elif model_type is ModelType.CNN:
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', name='extractor',
                         input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation=LAST_ACTIVATION))

    elif model_type is ModelType.CNN_LSTM:
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                         input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Dense(1, activation=LAST_ACTIVATION))

    elif model_type is ModelType.LSTM_AUTOENCODER:
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', name='extractor',
                         input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dropout(0.3))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(Xtrain.shape[1], Xtrain.shape[2]))))
        model.add(RepeatVector(10))
        model.add(Bidirectional(LSTM(50, activation='relu')))
        model.add(Dense(1))

    elif model_type is ModelType.DEEP_CNN:
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'),
                                  input_shape=(None, Xtrain.shape[1], Xtrain.shape[2])))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='softmax'))

    elif model_type is ModelType.GRU:
        model = Sequential()
        model.add(GRU(75, return_sequences=True, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(GRU(units=30, return_sequences=True))
        model.add(GRU(units=30))
        model.add(Dense(units=1, activation=LAST_ACTIVATION))

    elif model_type is ModelType.GRU_CNN:
        inp_seq = Input(shape=(Xtrain.shape[1], Xtrain.shape[2]))
        x = Bidirectional(GRU(100, return_sequences=True))(inp_seq)
        x = AveragePooling1D(2)(x)
        x = Conv1D(100, 3, activation='relu', padding='same',
                   name='extractor')(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.5)(x)

        out = Dense(1, activation=LAST_ACTIVATION)(x)

        model = Model(inp_seq, out)

    else:
        print("ERROR ", model_type)
        return None

    model.compile(loss=loss_func, optimizer=opt)
    # fit network
    if summary:
        model.summary()
    return model


def fit(Xtrain, Ytrain, Xtest, Ytest, model, epochs=20, batch_size=32, graph=False, binary=False, pos_weight=1,
        verbose=1):
    min_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    checkpoint_es = ModelCheckpoint(
        filepath='C:\\Users\\nitza\\Local\\WWTP\\models\\model.{epoch:02d}-{val_loss:.5f}.h5')
    nan_es = TerminateOnNaN()

    history = model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xtest, Ytest),
                        verbose=verbose, shuffle=True,
                        use_multiprocessing=False, callbacks=[nan_es,min_es])

    # plot history
    if graph:
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    return model


def evaluate(model, Xtest, Ytest, scalers, binary=False):
    Yhat = model.predict(Xtest)
    if not binary:
        Yhat = scalers[-1].inverse_transform(Yhat)
        Ytest = scalers[-1].inverse_transform(Ytest)
    return Yhat, Ytest

