# 0827 진행 상황
- BG

- HS

- JH

- JW
    - lstm코드만 추가 (xgboost 필요)
    - 추론 때 사용할 scaler를 저장해 두면 좋겠다는 생각 [22년이 아닌 23년, 즉 새로운 데이터들로 추론을 매일 하도록 할 것이라면 필요 ]
- SM

- TW

# 공통
- 추가한 라이브러리
    - from os.path import join as opj # 경로 설정을 위해서
- configuration 파일(.yml) 명명 규칙
    - 자산명_모델명_학습추론여부.yml (ex. ks_lstm_train.yml -> 코스피를 lstm으로 학습하는 설정)
- configuration 파일 내 경로는 stage1이 시작 경로로 되어 있다.
- configuration 파일 내 task_name == index_name..
# JW


- import 해볼 때는 load_data처럼 class도 불러올 수 있구나!