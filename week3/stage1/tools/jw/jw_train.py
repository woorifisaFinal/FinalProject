# -*- coding: utf-8 -*-
"""JW_KS_LSTM_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SH5BSpLNYOY3gV4sOSo6eLCm6wXRz6oy

Standard Scale -> test 제외 후 각각
"""

## 기존 라이브러리
import pandas as pd

## 데이터 불러오기 위한 class
from stage1.data import DataPreprocess

## 모델 불러오기 위한 함수
from stage1.models import create_jw_model

from os.path import join as opj

## 환경설정 관련 라이브러리
import argparse
import importlib
import yaml
from types import SimpleNamespace
import sys

# pyyaml docs : https://pyyaml.org/wiki/PyYAMLDocumentation
# Ref : https://github.com/ybabakhin/kaggle-feedback-effectiveness-1st-place-solution/blob/main/train.py
### 예시로 살펴볼 수 있는 yaml 파일들 
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/configs/cls/cls_mv3.yml
# Ref : https://github.com/ybabakhin/kaggle-feedback-effectiveness-1st-place-solution/blob/main/yaml/awesome-rose-ff.yaml

## 환경설정 얻어오기 위해 인자 받기
parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")

parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
# print(cfg)



def train(cfg):
    """
    데이터 -> 모델 -> 학습 & 검증 -> (모델) 저장
    """
    ##### 데이터 불러오기

    trainX, trainY, valX, valY, testX, testY  = DataPreprocess(cfg).load_data()

    ##### 모델 불러오기

    model = create_jw_model(trainX, trainY)

    ##### 학습
    
    validation_data = (valX,valY)

    # import matplotlib.pyplot as plt
    try:
        model.load_weights(opj(cfg.base.output_dir, 'lstm_weights_last.h5'))
        print("Loaded model weights from disk")
    except:
        # Fit the model
        history = model.fit(trainX, trainY, epochs=400, batch_size=4, validation_data=validation_data,
                        verbose=1) #validation - 과적합 방지 & loss 가 작을수록 좋은 모델이니 그것을 선택함

        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()

    # prediction
    prediction = model.predict(testX)
    print(prediction.shape, testY.shape)

    # 추가.. 예측 결과 저장 (stage2에서 쓰도록)
    import pickle
    with open(opj(cfg.base.output_dir, "prediction_22.pkl"), 'rb') as f:
        pickle.dump(prediction, f)
        
    # y_pred = np.squeeze(prediction)

    # testY_original = np.squeeze(testY)

    # # plotting
    # plt.figure(figsize=(14, 5))

    # # plot original 'returns' prices
    # plt.plot(dates, original_returns, color='green', label='Original Returns')

    # # plot actual vs predicted
    # plt.plot(test_dates[seq_len:], testY_original, color='blue', label='Actual Returns')
    # plt.plot(test_dates[seq_len:], y_pred, color='red', linestyle='--', label='Predicted Returns')
    # plt.xlabel('Date')
    # plt.ylabel('Returns')
    # plt.title('Original, Actual and Predicted Returns')
    # plt.legend()
    # plt.show()

    # len(y_pred)

    # # Calculate the start and end indices for the zoomed plot
    # zoom_start = len(test_dates) - 50
    # zoom_end = len(test_dates)

    # # Create the zoomed plot
    # plt.figure(figsize=(14, 5))

    # # Adjust the start index for the testY_original and y_pred arrays
    # adjusted_start = zoom_start - seq_len

    # plt.plot(test_dates[zoom_start:zoom_end],
    #         testY_original[adjusted_start:zoom_end - zoom_start + adjusted_start],
    #         color='blue',
    #         label='Actual Returns')

    # plt.plot(test_dates[zoom_start:zoom_end],
    #         y_pred[adjusted_start:zoom_end - zoom_start + adjusted_start ],
    #         color='red',
    #         linestyle='--',
    #         label='Predicted Returns')

    # plt.xlabel('Date')
    # plt.ylabel('Returns')
    # plt.title('Zoomed In Actual vs Predicted Returns')
    # plt.legend()
    # plt.show()

    # from sklearn.metrics import mean_squared_error
    # rmse = np.sqrt(mean_squared_error(testY_original, y_pred))
    # print("RMSE: %f" % (rmse))
    # # RMSE: 0.160966 ..?gg
    # # RMSE: 0.139228 옹예

    # from sklearn.metrics import mean_absolute_error
    # mae = mean_absolute_error(testY_original, y_pred)
    # print("MAE: %f" % (mae))

    # model.save('ks_lstm_last.h5')


    
    
    
    ##### 검증


if __name__=="__main__":
    train(cfg)