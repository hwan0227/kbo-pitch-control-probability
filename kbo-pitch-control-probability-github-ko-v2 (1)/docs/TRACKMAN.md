# TrackMan 통합

## 목적

TrackMan을 평가 중 현재 공의 비밀 정보로 사용한 것이 아니라, **과거에 관측된 투수의 물리 프로필**로 사용했다.

## 매칭

외부 선수 명단을 사용하지 않고 공식 데이터의 시즌, 경기 상태, 투수·타자 단서로 ID를 연결했다. 정밀 투구 키의 match precision은 약 99.99%였지만 엄격한 1:1 coverage는 약 70.07%였고, 17.28%는 모호했다. 그래서 row-level join을 무리하게 배포하기보다 다음과 같은 계층형 lookup을 구축했다.

1. strict exact mapping
2. 합의된 선수 매핑
3. 시즌 이전까지의 pitcher profile
4. 정보가 없을 때 league/pitch-type prior

## 만든 피처

- release speed, spin rate
- induced vertical/horizontal break
- release height/side, extension
- fastball/breaking/offspeed mix
- pitch-type separation
- 투수 내 평균과 표준편차
- 최근 변화량과 career 대비 shift

## 중요한 실패와 수정

초기 TrackMan 원본 행 순서가 진짜 시간순이라는 가정이 틀렸고, 이 순서에 의존한 rolling block은 폐기했다. 이후에는 날짜·경기 키로 재정렬할 수 있는 정보와 strict-prior aggregate만 사용했다.

평균 물리 프로필은 기존 선수/as-of 피처와 많이 겹쳤다. 반면 투수 내 산포는 기존 80열로 잘 재구성되지 않아 새로운 정보로 남았다. 이 때문에 TrackMan에서 얻은 실제 성과는 “더 많은 물리 수치”보다 **배달 동작의 반복성**을 나타내는 dispersion에서 나왔다.

## 규정 준수

- 평가 행 간 TrackMan/as-of 차분 금지
- 2025 행에는 2024까지 확정된 lookup만 사용
- 매핑의 근거는 공식 제공 데이터로 제한
- 모호한 매칭은 강제하지 않고 prior로 fallback
