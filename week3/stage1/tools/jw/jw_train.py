import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from pickle import dump


def createlstm():
# 데이터 불러오기
    raw_train = pd.read_csv("adj_raw_train.csv")
    raw_train['date'] = pd.to_datetime(raw_train['date'])
    dates = pd.to_datetime(raw_train['date'])



    target_year=2021
    train = raw_train[raw_train['date'].dt.year<target_year]
    target_year=2021
    validation = raw_train[raw_train['date'].dt.year==target_year]
    target_year=2022
    test = raw_train[raw_train['date'].dt.year==target_year]
    raw_train.set_index("date",inplace=True)


    # 데이터 내보내기
    # train.to_csv("jw_train.csv")
    # validation.to_csv("jw_validation.csv")
    # test.to_csv("jw_test.csv")



    # train 데이터 전처리
    cols = list(raw_train)[0:9]
    stock_data = raw_train[cols].astype(float)

    n_train = len(train)
    n_validation = n_train + len(validation)
    n_test = n_validation


    scaler = StandardScaler()
    scaler = scaler.fit(stock_data[:n_validation])
    stock_data_scaled = scaler.transform(stock_data[:n_validation])
    stock_data_scaled_test = scaler.transform(stock_data[n_validation:])


    dump(scaler, open('./lstm_scaler.pkl', 'wb'))

    stock_data_target = raw_train[["target"]]


    # split- X
    train_data_scaled = stock_data_scaled[0: n_train]
    train_dates = dates[0: n_train]

    val_data_scaled = stock_data_scaled[n_train: n_validation]
    val_dates = dates[n_train: n_validation]

    test_data_scaled = stock_data_scaled_test
    test_dates = dates[n_test:]

    # split- y
    train_data_test_scaled = stock_data_target[0: n_train]
    train_dates = dates[0: n_train]

    val_data_test_scaled = stock_data_target[n_train: n_validation]
    val_dates = dates[n_train: n_validation]

    test_data_test_scaled = stock_data_target[n_test:]
    test_dates = dates[n_test:]




    # data reformatting for LSTM
    pred_days = 1  
    seq_len = 10  
    input_dim = 10  

    trainX = []
    trainY = []
    valX = []
    valY = []
    testX = []
    testY = []

    for i in range(seq_len, n_train-pred_days +1):
        trainX.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
        trainY.append(train_data_test_scaled[i + pred_days - 1:i + pred_days].values)

    for i in range(seq_len, len(val_data_scaled)-pred_days +1):
        valX.append(val_data_scaled[i - seq_len:i, 0:val_data_scaled.shape[1]])
        valY.append(val_data_test_scaled[i + pred_days - 1:i + pred_days].values)

    for i in range(seq_len, len(test_data_scaled)-pred_days +1):
        testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
        testY.append(test_data_test_scaled[i + pred_days - 1:i + pred_days].values)

    trainX, trainY = np.array(trainX), np.array(trainY)
    valX, valY = np.array(valX), np.array(valY)
    testX, testY = np.array(testX), np.array(testY)
    validation_data = (valX,valY)



    model = Sequential()
    model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]),
                return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(trainY.shape[1]))


    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')



    model.fit(trainX, trainY, epochs=100, batch_size=4, validation_data=validation_data, verbose=2)


    model.save('jw_lstm_model')

createlstm()