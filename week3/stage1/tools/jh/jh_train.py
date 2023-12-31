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
from stage1.models import build_rnn_model

from os.path import join as opj
from stage1.utils import get_logger
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

logger = get_logger(cfg.base)

def train(cfg):
    """
    데이터 -> 모델 -> 학습 & 검증 -> (모델) 저장
    """
    ##### 데이터 불러오기
    
    X_train, X_valid, y_train, y_valid = DataPreprocess(cfg).load_data(logger)

    ##### 모델 불러오기

    model = build_rnn_model(cfg)
    logger.info(model.summary())

    ##### 학습
    
    h = model.fit(X_train,y_train,
                    validation_data = (X_valid,y_valid),
                    # sample_weight = np.tile(w,GRP),
                    batch_size=4, epochs=10, verbose=2)

    ### 모델 저장
    model.save_weights(f"{cfg.base.stage_name}_{cfg.base.model_name}_{cfg.base.exp_name}.h5")
    # model.load_weights(INFER_FROM_PATH + f'GRU_f{fold}_v{VER}.h5')



if __name__=="__main__":
    
    train(cfg, logger)