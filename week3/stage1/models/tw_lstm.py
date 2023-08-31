import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
def bond_short():
# 데이터 로드 및 전처리
  data = pd.read_csv('/content/3년국채 데이터17_22.csv', encoding='cp949')
  data['날짜'] = pd.to_datetime(data['날짜'])
  data = data.sort_values(by='날짜')
  data = data[['날짜', '종가', '시가', '저가', '변동률', '수익률']]
  data['종가'] = data['종가'].str.replace(',', '').astype(float)
  data['시가'] = data['시가'].str.replace(',', '').astype(float)
  data['저가'] = data['저가'].str.replace(',', '').astype(float)
  data['변동률'] = data['변동률'].str.replace('%', '').astype(float) / 100.0
  data['수익률'] = data['수익률'].str.replace('%', '').astype(float) / 100.0

  # 데이터 스케일링
  scaler_closing = MinMaxScaler()
  scaler_opening = MinMaxScaler()
  scaler_low = MinMaxScaler()
  scaler_volatility = MinMaxScaler()
  scaler_return = MinMaxScaler()

  data['종가'] = scaler_closing.fit_transform(data[['종가']])
  data['시가'] = scaler_opening.fit_transform(data[['시가']])
  data['저가'] = scaler_low.fit_transform(data[['저가']])
  data['변동률'] = scaler_volatility.fit_transform(data[['변동률']])
  data['수익률'] = scaler_return.fit_transform(data[['수익률']])

  # 시계열 데이터 생성 함수 수정
  def create_sequence_data(data, sequence_length):
      sequences = []
      for i in range(len(data) - sequence_length + 1):
          sequence = data.iloc[i:i + sequence_length][['종가', '시가', '저가', '변동률']].values  # 수익률 제외
          if i + sequence_length < len(data):
              target = data.iloc[i + sequence_length]['수익률']  # 수익률을 타겟으로 변경
              sequences.append((sequence, target))
      return sequences

  sequence_length = 3  # 시퀀스 길이 설정
  train_data = data[(data['날짜'].dt.year >= 2017) & (data['날짜'].dt.year <= 2021)]
  train_sequences = create_sequence_data(train_data, sequence_length)
  X_train = np.array([sequence for sequence, target in train_sequences])
  y_train = np.array([target for sequence, target in train_sequences])

  model = Sequential([
  LSTM(50, activation='relu', input_shape=(sequence_length, 4)),# 4개 변수를 입력으로 설정
  Dense(1)
  ])

  model.compile(optimizer='adam', loss='mse')

  # 모델 학습
  model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

  # 2022년 데이터로 예측 수행
  test_data_2022 = data[data['날짜'].dt.year == 2022]
  test_sequences_2022 = create_sequence_data(test_data_2022, sequence_length)
  X_test_2022 = np.array([sequence for sequence, target in test_sequences_2022])
  y_test_2022 = np.array([target for sequence, target in test_sequences_2022])
  predictions_2022 = model.predict(X_test_2022)

  # 예측 결과 시각화
  plt.figure(figsize=(12, 6))
  plt.plot(np.arange(len(y_test_2022)), y_test_2022, label='Real data (2022)')
  plt.plot(np.arange(len(y_test_2022)), predictions_2022, label='Forecast data (2022)')
  plt.xlabel('days')
  plt.ylabel('return')
  plt.title('LSTM Forecast data for 2022')
  plt.legend()
  plt.xticks(rotation=45)
  plt.show()

  # RMSE 계산
  rmse_2022 = np.sqrt(mean_squared_error(y_test_2022, predictions_2022))
  print(f'Root Mean Squared Error for 2022: {rmse_2022}')


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
def bond_long():
# 데이터 로드 및 전처리
  data = pd.read_csv('/content/한국10 병합.csv', encoding='cp949')
  data['날짜'] = pd.to_datetime(data['날짜'])
  data = data.sort_values(by='날짜')
  data = data[['날짜', '종가', '시가', '저가', '변동률', '수익률']]
#   data['종가'] = data['종가'].str.replace(',', '').astype(float)
#   data['시가'] = data['시가'].str.replace(',', '').astype(float)
#   data['저가'] = data['저가'].str.replace(',', '').astype(float)
#   data['변동률'] = data['변동률'].str.replace('%', '').astype(float) / 100.0
#   data['수익률'] = data['수익률'].str.replace('%', '').astype(float) / 100.0

  # 데이터 스케일링
  scaler_closing = MinMaxScaler()
  scaler_opening = MinMaxScaler()
  scaler_low = MinMaxScaler()
  scaler_volatility = MinMaxScaler()
  scaler_return = MinMaxScaler()

  data['종가'] = scaler_closing.fit_transform(data[['종가']])
  data['시가'] = scaler_opening.fit_transform(data[['시가']])
  data['저가'] = scaler_low.fit_transform(data[['저가']])
  data['변동률'] = scaler_volatility.fit_transform(data[['변동률']])
  data['수익률'] = scaler_return.fit_transform(data[['수익률']])

  # 시계열 데이터 생성 함수 수정
  def create_sequence_data(data, sequence_length):
      sequences = []
      for i in range(len(data) - sequence_length + 1):
          sequence = data.iloc[i:i + sequence_length][['종가', '시가', '저가', '변동률']].values  # 수익률 제외
          if i + sequence_length < len(data):
              target = data.iloc[i + sequence_length]['수익률']  # 수익률을 타겟으로 변경
              sequences.append((sequence, target))
      return sequences

  sequence_length = 3  # 시퀀스 길이 설정
  train_data = data[(data['날짜'].dt.year >= 2017) & (data['날짜'].dt.year <= 2021)]
  train_sequences = create_sequence_data(train_data, sequence_length)
  X_train = np.array([sequence for sequence, target in train_sequences])
  y_train = np.array([target for sequence, target in train_sequences])

  model = Sequential([
  LSTM(50, activation='relu', input_shape=(sequence_length, 4)),# 4개 변수를 입력으로 설정
  Dense(1)
  ])

  model.compile(optimizer='adam', loss='mse')

  # 모델 학습
  model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

  # 2022년 데이터로 예측 수행
  test_data_2022 = data[data['날짜'].dt.year == 2022]
  test_sequences_2022 = create_sequence_data(test_data_2022, sequence_length)
  X_test_2022 = np.array([sequence for sequence, target in test_sequences_2022])
  y_test_2022 = np.array([target for sequence, target in test_sequences_2022])
  predictions_2022 = model.predict(X_test_2022)

  # 예측 결과 시각화
  plt.figure(figsize=(12, 6))
  plt.plot(np.arange(len(y_test_2022)), y_test_2022, label='Real data (2022)')
  plt.plot(np.arange(len(y_test_2022)), predictions_2022, label='Forecast data (2022)')
  plt.xlabel('days')
  plt.ylabel('return')
  plt.title('LSTM Forecast data for 2022')
  plt.legend()
  plt.xticks(rotation=45)
  plt.show()

  # RMSE 계산
  rmse_2022 = np.sqrt(mean_squared_error(y_test_2022, predictions_2022))
  print(f'Root Mean Squared Error for 2022: {rmse_2022}')


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
def us_bond_short():
# 데이터 로드 및 전처리
  data = pd.read_csv('/content/미국3 병합.csv', encoding='cp949')
  data['날짜'] = pd.to_datetime(data['날짜'])
  data = data.sort_values(by='날짜')
  data = data[['날짜', '종가', '시가', '저가', '변동률', '수익률']]
#   data['종가'] = data['종가'].str.replace(',', '').astype(float)
#   data['시가'] = data['시가'].str.replace(',', '').astype(float)
#   data['저가'] = data['저가'].str.replace(',', '').astype(float)
#   data['변동률'] = data['변동률'].str.replace('%', '').astype(float) / 100.0
#   data['수익률'] = data['수익률'].str.replace('%', '').astype(float) / 100.0

  # 데이터 스케일링
  scaler_closing = MinMaxScaler()
  scaler_opening = MinMaxScaler()
  scaler_low = MinMaxScaler()
  scaler_volatility = MinMaxScaler()
  scaler_return = MinMaxScaler()

  data['종가'] = scaler_closing.fit_transform(data[['종가']])
  data['시가'] = scaler_opening.fit_transform(data[['시가']])
  data['저가'] = scaler_low.fit_transform(data[['저가']])
  data['변동률'] = scaler_volatility.fit_transform(data[['변동률']])
  data['수익률'] = scaler_return.fit_transform(data[['수익률']])

  # 시계열 데이터 생성 함수 수정
  def create_sequence_data(data, sequence_length):
      sequences = []
      for i in range(len(data) - sequence_length + 1):
          sequence = data.iloc[i:i + sequence_length][['종가', '시가', '저가', '변동률']].values  # 수익률 제외
          if i + sequence_length < len(data):
              target = data.iloc[i + sequence_length]['수익률']  # 수익률을 타겟으로 변경
              sequences.append((sequence, target))
      return sequences

  sequence_length = 3  # 시퀀스 길이 설정
  train_data = data[(data['날짜'].dt.year >= 2017) & (data['날짜'].dt.year <= 2021)]
  train_sequences = create_sequence_data(train_data, sequence_length)
  X_train = np.array([sequence for sequence, target in train_sequences])
  y_train = np.array([target for sequence, target in train_sequences])

  model = Sequential([
  LSTM(50, activation='relu', input_shape=(sequence_length, 4)),# 4개 변수를 입력으로 설정
  Dense(1)
  ])

  model.compile(optimizer='adam', loss='mse')

  # 모델 학습
  model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

  # 2022년 데이터로 예측 수행
  test_data_2022 = data[data['날짜'].dt.year == 2022]
  test_sequences_2022 = create_sequence_data(test_data_2022, sequence_length)
  X_test_2022 = np.array([sequence for sequence, target in test_sequences_2022])
  y_test_2022 = np.array([target for sequence, target in test_sequences_2022])
  predictions_2022 = model.predict(X_test_2022)

  # 예측 결과 시각화
  plt.figure(figsize=(12, 6))
  plt.plot(np.arange(len(y_test_2022)), y_test_2022, label='Real data (2022)')
  plt.plot(np.arange(len(y_test_2022)), predictions_2022, label='Forecast data (2022)')
  plt.xlabel('days')
  plt.ylabel('return')
  plt.title('LSTM Forecast data for 2022')
  plt.legend()
  plt.xticks(rotation=45)
  plt.show()

  # RMSE 계산
  rmse_2022 = np.sqrt(mean_squared_error(y_test_2022, predictions_2022))
  print(f'Root Mean Squared Error for 2022: {rmse_2022}')

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
def us_bond_long():
# 데이터 로드 및 전처리
  data = pd.read_csv('/content/미국10 병합.csv', encoding='cp949')
  data['날짜'] = pd.to_datetime(data['날짜'])
  data = data.sort_values(by='날짜')
  data = data[['날짜', '종가', '시가', '저가', '변동률', '수익률']]
#   data['종가'] = data['종가'].str.replace(',', '').astype(float)
#   data['시가'] = data['시가'].str.replace(',', '').astype(float)
#   data['저가'] = data['저가'].str.replace(',', '').astype(float)
#   data['변동률'] = data['변동률'].str.replace('%', '').astype(float) / 100.0
#   data['수익률'] = data['수익률'].str.replace('%', '').astype(float) / 100.0

  # 데이터 스케일링
  scaler_closing = MinMaxScaler()
  scaler_opening = MinMaxScaler()
  scaler_low = MinMaxScaler()
  scaler_volatility = MinMaxScaler()
  scaler_return = MinMaxScaler()

  data['종가'] = scaler_closing.fit_transform(data[['종가']])
  data['시가'] = scaler_opening.fit_transform(data[['시가']])
  data['저가'] = scaler_low.fit_transform(data[['저가']])
  data['변동률'] = scaler_volatility.fit_transform(data[['변동률']])
  data['수익률'] = scaler_return.fit_transform(data[['수익률']])

  # 시계열 데이터 생성 함수 수정
  def create_sequence_data(data, sequence_length):
      sequences = []
      for i in range(len(data) - sequence_length + 1):
          sequence = data.iloc[i:i + sequence_length][['종가', '시가', '저가', '변동률']].values  # 수익률 제외
          if i + sequence_length < len(data):
              target = data.iloc[i + sequence_length]['수익률']  # 수익률을 타겟으로 변경
              sequences.append((sequence, target))
      return sequences

  sequence_length = 3  # 시퀀스 길이 설정
  train_data = data[(data['날짜'].dt.year >= 2017) & (data['날짜'].dt.year <= 2021)]
  train_sequences = create_sequence_data(train_data, sequence_length)
  X_train = np.array([sequence for sequence, target in train_sequences])
  y_train = np.array([target for sequence, target in train_sequences])

  model = Sequential([
  LSTM(50, activation='relu', input_shape=(sequence_length, 4)),# 4개 변수를 입력으로 설정
  Dense(1)
  ])

  model.compile(optimizer='adam', loss='mse')

  # 모델 학습
  model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

  # 2022년 데이터로 예측 수행
  test_data_2022 = data[data['날짜'].dt.year == 2022]
  test_sequences_2022 = create_sequence_data(test_data_2022, sequence_length)
  X_test_2022 = np.array([sequence for sequence, target in test_sequences_2022])
  y_test_2022 = np.array([target for sequence, target in test_sequences_2022])
  predictions_2022 = model.predict(X_test_2022)

  # 예측 결과 시각화
  plt.figure(figsize=(12, 6))
  plt.plot(np.arange(len(y_test_2022)), y_test_2022, label='Real data (2022)')
  plt.plot(np.arange(len(y_test_2022)), predictions_2022, label='Forecast data (2022)')
  plt.xlabel('days')
  plt.ylabel('return')
  plt.title('LSTM Forecast data for 2022')
  plt.legend()
  plt.xticks(rotation=45)
  plt.show()

  # RMSE 계산
#   rmse_2022 = np.sqrt(mean_squared_error(y_test_2022, predictions_2022))
#   print(f'Root Mean Squared Error for 2022: {rmse_2022}')

