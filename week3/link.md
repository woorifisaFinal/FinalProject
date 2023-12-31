회의록 (정리본), 지우 참고자료 :  https://chatgptlearningday.notion.site/5-3-70588129d9ba4f10bfeab8e05bd98db1?pvs=4  
수민 참고자료: 
    xgboost: https://colab.research.google.com/drive/1oF5twMjJwV_Ey5A60iWjf5bCAul8dnKH?usp=sharing
    lstm : https://colab.research.google.com/drive/1fAjFwXwpApOyzjNMCe7UiKehF57Y1zRo?usp=sharing    
주혁 참고자료:  
병근 참고자료: https://slash-crest-e4f.notion.site/5296eb979b2b438497b68dc2d65be0c2?pvs=4  
태완 참고자료: https://colab.research.google.com/drive/1QIDJP9VbvVJhUWpx0hDS7vBoLBEDBpVY?usp=sharing  
하성 참고자료:   


# 0821

- AWS 인프라 설정 → 수민, 하성
- 주가예측 마무리
    - 백엔드 연결 → 주혁, 지우
- 포트폴리오 알고리즘
    - 논문 스터디 → 병근, 태완
- 수요일 오후: 아웃라인, to do, 대략의 plan
- 금요일 오후: 설문조사 돌리기
- 시계열/예측 모델 토의
  - Validation 까지 재학습 (주기단위, 리밸런싱 주기와 같음)
  - 포트폴리오 LSTM 사용

```
2주차 결과 및 보완점

#추가적으로 할일
- 상관관계가 높은 변수 제외  
다중공선성 문제를 방지
- 종속변수 정규화(xgboost) -> 정규화가 되어있긴 하지만 다른 정규화 방식을 해주는것도 좋음
- 앙상블 (참고: https://knowallworld.tistory.com/399)
- 3개월 후
https://dacon.io/en/competitions/official/236117/codeshare/8680?page=1&dtype=recent&ptype&fType
- 데이터셋 변경

[오수민] [오후 2:13] *8/21*
보완할 부분)
lstm의 예측 그래프는 곧잘 실제 값 그래프를 따라가는 데 예측 값 자체는 실제값과 차이가 크다 

[이주혁] [오후 2:16] 전체적인 틀 (Baseline) 구축이 목표
configuration은 Yaml파일로 정리되어 있고
index_name만 변경해주면 
세 국가 모두에 대해 예측할 수 있다.
model_name을 GRU이나 LSTM으로 바꾸면 변경가능하다.
lookback window : 10 ->이전 10일을 가지고 예측
lookahead 62 -> 62번째 다음 행(3개월 후가 공휴일일 경우 대비, 17-1-2에 3개월 후가 (학습 데이터프레임 기준)62번째에 존재하므로 62로 선정

EDA : Volume이 0인 것들이 있다.

Make Data에서 시계열 데이터로 만든다.
Metric에 smape, rmse코드 존재

RUN에 데이터 받아오기-데이터 처리-학습 등 코드가 모여 있음.

성능이 좋지 않은데, 수민님 코드 따라서 다음날 예측으로 하면 어느정도 추세를 따라감을 알 수 있다.
영점조절(Multiplier 활용, -> 특정 float값)하면 된다.
즉, 데이터 범위는 중요하지 않고 추세...

[김태완] [오후 2:17] 보완
XGB모델에서 22년 3월 이후의 데이터가 직선으로 나오는 부분을 수정해야됨.
LSTM , XGB 모델의 베이스라인이 제대로 작동하는지 확인.
기본적으로 추가할만한 칼럼들이 있는지 궁금함.

[최하성] [오후 2:17] xgboost는 성능이 과도하게 높게 나와서 parameter tuning을 다시 해봐야 할 것 같음. 
ensemble 해볼 예정. 반대로 ARIMA의 성능은 낮게 나왔는데 그 원인을 알아봐야 할 것 같고 다음 시간까지 LSTM 돌려볼 예정. 
kaggle에서 양식을 참고하고자 함. Train set과 validation set은 17~20년치를 train set으로 21년치를 validation set으로 다시 학습시킬 예정.
 마지막으로 이번에는 모델 돌리는 과정에서 logger를 남겨볼 생각임.

[문병근] [오후 2:17] 소감
1. 조금 더 상관관계가 높은 변수를 찾으면 좋겠지만 찾기가 여간 쉬운일이 아님.
2. LSTM에 대한 이해도가 아직 부족하다 => 좀 더 공부 필요 내것으로 만들기
3. XGB 파라미터 변경해서 성능을 높여 볼 필요가 있어보인다.
4. 수익률과 종가에 관한 고찰
5. 깃허브 사용법에 대해 좀더 능숙해져야 할 것같음.
6. Train, Val, Test 제대로 적용해봐야함.
7. 어떻게 하면 LSTM 이라는 것을 내 자료에 맞게 100프로 활용 할 수 있는가
8. 데이터 전처리에 시간이 많이 소요되었다.
9. 사용한 모델에 대해 이해도 높이기.
```

