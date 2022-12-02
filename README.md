# 1. 음원가사 감정 분류 기준

## 1. Robert Plutchik 감정모델(Electra)

</br>
<p align="center"><img src=./img/Plutchik-wheel.png width=50% title="Robert Plutchik's Emotional Wheel"></p>

- 감정 분류
  - joy
  - trust
  - fear
  - surprise
  - sadness
  - disgust
  - anger
  - anticipation

### A. Base

- 감정 분류 그대로 사용
- 정확도 약 40%

### B. + Neutrality

- 기존 감정 분류에 중립 감정 추가
- 정확도 약 44%

### C. - Anticipation

- 기존 감정 분류에 중립 감정을 추가하고 ‘기대’ 감정 제거
- 정확도 약 47%

## 3. 긍정, 중립, 부정 3분류

- AI hub 데이터를 사용하여 긍정, 부정, 중립 감정으로 라벨링
- 정확도 70%

### A. 부정 감정 세부 분류

- 긍정 + 중립 데이터를 제외한 데이터로 모델 학습 및 예측
- angry, dislike(dislike + contempt), fear, sad, surprise 5가지로 세부 분류

## 2. Paul Ekman 감정모델(Electra)

</br>
<p align="center"><img src=./img/paul_ekman.png width=50% title="Paul Ekman's Emotional Model"></p>

- 감정 분류
  - angry
  - fear
  - happy
  - sad
  - surprsie
  - contempt

### A. AI hub 데이터 사용

- 일상 대화 데이터 약 8만개 사용
- 기존 분류에 dislike, 중립 감정 추가
- 정확도 약 50%

### B. - contempt

- contempt 감정 → dislike로 분류
- 정확도 약 54%

### C. + complex

- 복합 감정 태그 추가
- 정확도 약 50%

# 2. 음원가사 전처리

### A. 어절 단위 120개 슬라이싱

- 학습할 수 있는 최대 토큰 개수
- Electra Tokenizer가 자동으로 토큰화하기 때문에 토큰 개수가 120개 보다 많아져서 문장이 잘리는 문제 확인

### B. 어절 단위 60개 슬라이싱

- 문장이 잘리는 문제를 해결하기 위해 반토막 냄
- 문장이 길어 한 문장에 여러 감정이 포함되는 문제가 생김
- 슬라이싱 후 남은 짧은 문장들이 발생

### C. 어절 단위 20개 슬라이싱

- 한 문장에 여러 감정이 포함되는 문제 해결
- 슬라이싱 후 남은 짧은 문장을 그 전 문장에 합침
